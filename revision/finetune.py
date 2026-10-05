"""
Step 3: LoRA fine-tuning on the TRAIN split only, one shared configuration for every model (R1-C1, R1-C11).

    python revision/finetune.py --model gemma3-12b --split-dir revision/splits/main \
        --index-dir revision/index/bge-m3 --format reasoning_first --rag \
        --out revision/models/gemma3-12b__reasoning_first

With --rag, each training prompt embeds the k nearest TRAIN neighbors (never the
article itself or any member of its duplicate cluster). --rag-dropout (default 0.5)
leaves the reference block out of that share of prompts, so ONE adapter is valid both
with and without retrieval at inference: the Non-RAG vs RAG comparison then differs
only in the prompt, not in the weights.

Loss is computed on the answer tokens only (prompt tokens are masked). Validation loss
on a fixed subset of val.jsonl selects the best checkpoint.
test.jsonl and test_xsource.jsonl are never opened by this script.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Shared configuration (paper: "standardized comparison"). Override only with a
# documented reason; every override is written to run_config.json.
SHARED = dict(lora_r=16, lora_alpha=32, lora_dropout=0.0, learning_rate=2e-4, epochs=2,
              per_device_batch=1, grad_accum=16, warmup_ratio=0.03, weight_decay=0.01,
              max_seq_length=2048, seed=3407)

MODELS = {
    "gemma3-270m": "unsloth/gemma-3-270m-it-bnb-4bit",
    "gemma3-1b": "unsloth/gemma-3-1b-it-bnb-4bit",
    "gemma3-4b": "unsloth/gemma-3-4b-it-bnb-4bit",
    "gemma3-12b": "unsloth/gemma-3-12b-it-bnb-4bit",
    "gemma2-9b": "unsloth/gemma-2-9b-it-bnb-4bit",
    "qwen3.5-2b": "unsloth/Qwen3.5-2B",
    "qwen3.5-9b": "unsloth/Qwen3.5-9B",
    "qwen3-8b": "unsloth/Qwen3-8B-bnb-4bit",
    "qwen2.5-14b": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    "llama3.2-1b": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "deepseek-r1-llama-8b": "unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit",
}


def build_examples(rows, tokenizer, fmt, use_evidence, retriever, k, rag_dropout, max_chars, nb_chars, seed):
    from prompts import apply_chat, build_prompt, build_target
    rng = random.Random(seed)
    out, n_rag = [], 0
    for r in rows:
        neighbors = None
        if retriever is not None and rng.random() >= rag_dropout:
            neighbors = retriever.search(retriever.query_vector(r), k, query_lang=r["lang"],
                                         exclude_ids=[r["id"]], exclude_clusters=[r["cluster_id"]])
            n_rag += 1
        prompt = apply_chat(tokenizer, build_prompt(r["text"], r["lang"], fmt, use_evidence, neighbors,
                                                    max_chars=max_chars, ref_chars=nb_chars))
        out.append({"prompt": prompt, "target": build_target(r, fmt, use_evidence) + tokenizer.eos_token})
    return out, n_rag


def tokenize(examples, tok, max_len):
    """Prompt tokens get label -100, so the loss covers the answer (signals + label) only.

    Tokenized exactly as infer.py does (chat template already holds BOS, so no special tokens
    are added). Examples longer than max_len are dropped and counted rather than truncated,
    because truncation would cut the answer.
    """
    rows, dropped = [], 0
    for ex in examples:
        p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
        t = tok(ex["target"], add_special_tokens=False)["input_ids"]
        if len(p) + len(t) > max_len:
            dropped += 1
            continue
        rows.append({"input_ids": p + t, "attention_mask": [1] * (len(p) + len(t)),
                     "labels": [-100] * len(p) + t})
    return rows, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=f"one of {list(MODELS)} or a Hugging Face id")
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="reasoning_first", choices=("label_only", "label_first",
                                                                     "reasoning_first", "assessment_only"))
    ap.add_argument("--rag", action="store_true", help="Train with retrieved TRAIN neighbors in the prompt")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rag-dropout", type=float, default=0.5)
    ap.add_argument("--evidence-file", default=None, help="JSONL {id, evidence} from generate_rationales.py")
    ap.add_argument("--max-chars", type=int, default=1500, help="Article characters in the prompt")
    ap.add_argument("--neighbor-chars", type=int, default=400)
    ap.add_argument("--max-train", type=int, default=None, help="Debug only: subsample train")
    ap.add_argument("--max-val", type=int, default=400,
                    help="Validation articles used for eval loss (fixed random subset; full val is slow to score)")
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--merge", action="store_true", help="Also save a merged 16-bit copy")
    ap.add_argument("--num-gpu", type=int, default=1)
    for key, val in SHARED.items():
        ap.add_argument("--" + key.replace("_", "-"), type=type(val), default=val)
    args = ap.parse_args()
    if args.rag and not args.index_dir:
        ap.error("--rag needs --index-dir")

    import _server_compat  # noqa: F401  (must precede unsloth on the cluster)
    from unsloth import FastLanguageModel
    import torch
    import math
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer, TrainingArguments
    from common import env_versions, read_jsonl, write_json
    from retrieval import Retriever

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.split_dir)
    train = read_jsonl(split_dir / "train.jsonl")
    val = read_jsonl(split_dir / "val.jsonl")
    if args.max_train:
        train = random.Random(0).sample(train, min(args.max_train, len(train)))
    if args.max_val:
        val = random.Random(1).sample(val, min(args.max_val, len(val)))
    use_evidence = bool(args.evidence_file)
    if use_evidence:
        ev = {r["id"]: r["evidence"] for r in read_jsonl(args.evidence_file)}
        missing = [r["id"] for r in train + val if r["id"] not in ev]
        assert not missing, f"{len(missing)} train/val records without evidence, e.g. {missing[:5]}"
        for r in train + val:
            r["evidence"] = ev[r["id"]]

    base = MODELS.get(args.model, args.model)
    overrides = {k: getattr(args, k) for k in SHARED if getattr(args, k) != SHARED[k]}
    if overrides:
        print(f"⚠️  Overriding the shared configuration: {overrides} (document this in the paper)")

    import os
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base, max_seq_length=args.max_seq_length, dtype=None, load_in_4bit=True,
        token=os.environ.get("HF_TOKEN"), device_map="balanced" if args.num_gpu > 1 else "cuda:0")
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=args.seed)
    tok = getattr(tokenizer, "tokenizer", tokenizer)  # Gemma 3 returns a processor

    retriever = Retriever(args.index_dir) if args.rag else None
    tr, n_rag_tr = build_examples(train, tokenizer, args.format, use_evidence, retriever, args.k,
                                  args.rag_dropout, args.max_chars, args.neighbor_chars, args.seed)
    va, n_rag_va = build_examples(val, tokenizer, args.format, use_evidence, retriever, args.k,
                                  args.rag_dropout, args.max_chars, args.neighbor_chars, args.seed + 1)
    with open(out / "example_prompt.txt", "w", encoding="utf-8") as f:
        f.write(tr[0]["prompt"] + tr[0]["target"])
    tr_tok, drop_tr = tokenize(tr, tok, args.max_seq_length)
    va_tok, drop_va = tokenize(va, tok, args.max_seq_length)
    lengths = [len(x["input_ids"]) for x in tr_tok]
    answer = [sum(l != -100 for l in x["labels"]) for x in tr_tok]
    print(f"train={len(tr_tok)} (with references: {n_rag_tr})  val={len(va_tok)} (with references: {n_rag_va})")
    print(f"tokens: max={max(lengths)} mean={sum(lengths) / len(lengths):.0f}; answer tokens mean="
          f"{sum(answer) / len(answer):.0f}; dropped as too long: train {drop_tr}, val {drop_va}")
    if drop_tr or drop_va:
        print("⚠️  Examples longer than max_seq_length were dropped. Lower --max-chars or raise --max-seq-length.")

    total_steps = math.ceil(len(tr_tok) / (args.per_device_batch * args.grad_accum)) * args.epochs
    trainer = Trainer(
        model=model,
        train_dataset=Dataset.from_list(tr_tok), eval_dataset=Dataset.from_list(va_tok),
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_batch, per_device_eval_batch_size=args.per_device_batch,
            gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
            learning_rate=args.learning_rate, warmup_steps=max(1, int(args.warmup_ratio * total_steps)),
            weight_decay=args.weight_decay,
            lr_scheduler_type="cosine", optim="adamw_8bit", seed=args.seed,
            fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10, eval_strategy="steps", eval_steps=args.eval_steps,
            save_strategy="steps", save_steps=args.eval_steps, save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            output_dir=str(out / "checkpoints"), report_to="none", dataloader_num_workers=0),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    stats = trainer.train()

    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    if args.merge:
        model.save_pretrained_merged(str(out) + "_merged_16bit", tokenizer, save_method="merged_16bit")

    write_json(out / "run_config.json", {
        "model_key": args.model, "base_model": base, "format": args.format, "use_evidence": use_evidence,
        "rag": args.rag, "index_dir": args.index_dir, "k": args.k, "rag_dropout": args.rag_dropout,
        "max_chars": args.max_chars, "neighbor_chars": args.neighbor_chars,
        "hyperparameters": {k: getattr(args, k) for k in SHARED}, "overrides": overrides,
        "effective_batch": args.per_device_batch * args.grad_accum,
        "n_train": len(tr_tok), "n_val": len(va_tok), "n_train_with_references": n_rag_tr,
        "n_dropped_too_long": {"train": drop_tr, "val": drop_va}, "loss": "answer tokens only",
        "split_dir": str(split_dir), "split_manifest": json.load(open(split_dir / "manifest.json"))["file_sha256"],
        "train_metrics": stats.metrics, "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric, "log_history": trainer.state.log_history,
        "versions": env_versions(),
    })
    print(f"✅ adapter saved to {out} (best eval_loss {trainer.state.best_metric})")


if __name__ == "__main__":
    main()
