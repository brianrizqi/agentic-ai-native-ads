"""
Step 4: inference on the locked test split. One script for every condition.

    # Non-RAG and RAG with the SAME adapter
    python revision/infer.py --model revision/models/gemma3-12b__reasoning_first \
        --test-file revision/splits/main/test.jsonl --out revision/results/gemma3-12b/norag.jsonl
    python revision/infer.py --model revision/models/gemma3-12b__reasoning_first --rag \
        --test-file revision/splits/main/test.jsonl --out revision/results/gemma3-12b/rag_k5.jsonl

Retrieval ablations (R1-C3): --rag-mode random|flipped|balanced, --hide-labels, --hide-text
Near-duplicate control (R1-C2): --max-sim 0.90
Cross-lingual retrieval (R1-C4): --neighbor-lang same|cross
Faithfulness (R1-C5): --intervene all_yes|all_no|flip_each (reasoning_first adapters only)
Zero-shot floor (Scenario 0): --model <HF base id> --format reasoning_first (no adapter)

Decoding is greedy (do_sample=False) for every model (R1 Q16). Invalid JSON is recorded as
label_source="invalid" and scored as wrong, never relabeled (R1 Q15). Re-running with the
same --out resumes from the last finished article.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SIGNALS, append_jsonl, label_from_signals, read_json, read_jsonl


def flip(name, value):
    return "tidak" if value == "ya" else "ya"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Adapter dir from finetune.py, or a HF base model id (zero-shot)")
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default=None, help="Default: from the adapter's run_config.json")
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--index-dir", default=None, help="Default: from run_config.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rag-mode", default="topk", choices=("topk", "balanced", "random", "flipped"))
    ap.add_argument("--hide-labels", action="store_true")
    ap.add_argument("--hide-text", action="store_true")
    ap.add_argument("--show-signals", action="store_true")
    ap.add_argument("--neighbor-lang", default="any", choices=("any", "same", "cross"))
    ap.add_argument("--max-sim", type=float, default=None)
    ap.add_argument("--intervene", default="none", choices=("none", "all_yes", "all_no", "flip_each"))
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--lang", default=None, help="Only evaluate one language (id|en)")
    ap.add_argument("--batch-size", type=int, default=8, help="Use 1 for the per-article latency table")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import _server_compat  # noqa: F401
    from unsloth import FastLanguageModel
    import numpy as np
    import torch
    from prompts import apply_chat, build_prompt, forced_signal_prefix, parse_output
    from retrieval import Retriever

    cfg_path = Path(args.model) / "run_config.json"
    cfg = read_json(cfg_path) if cfg_path.exists() else {}
    fmt = args.format or cfg.get("format")
    if not fmt:
        ap.error("--format is required for a base model without run_config.json")
    use_evidence = cfg.get("use_evidence", False)
    max_chars, ref_chars = cfg.get("max_chars", 1500), cfg.get("neighbor_chars", 400)
    max_new = args.max_new_tokens or {"label_only": 24}.get(fmt, 384 if use_evidence else 128)
    if args.intervene != "none" and fmt != "reasoning_first":
        ap.error("--intervene needs a reasoning_first model (signals must precede the label)")

    retriever = None
    if args.rag:
        index_dir = args.index_dir or cfg.get("index_dir")
        if not index_dir:
            ap.error("--rag needs --index-dir")
        retriever = Retriever(index_dir)
    rng = np.random.default_rng(args.seed)

    rows = read_jsonl(args.test_file)
    if args.lang:
        rows = [r for r in rows if r["lang"] == args.lang]
    if args.max_samples:
        rows = rows[: args.max_samples]  # test files are pre-shuffled with a fixed seed
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} articles, {len(done)} already done, {len(todo)} to run | format={fmt} rag={args.rag}")

    import os
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.model, max_seq_length=args.max_seq_length,
                                                         dtype=None, load_in_4bit=True, token=os.environ.get("HF_TOKEN"))
    FastLanguageModel.for_inference(model)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    condition = {"model": args.model, "format": fmt, "rag": args.rag, "k": args.k if args.rag else 0,
                 "rag_mode": args.rag_mode, "hide_labels": args.hide_labels, "hide_text": args.hide_text,
                 "show_signals": args.show_signals, "neighbor_lang": args.neighbor_lang, "max_sim": args.max_sim,
                 "intervene": args.intervene, "test_file": args.test_file, "decoding": "greedy",
                 "max_new_tokens": max_new, "batch_size": args.batch_size,
                 "index_model": retriever.info["model"] if retriever else None}

    @torch.no_grad()
    def generate(texts):
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        return [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True) for g in gen], \
            int(enc["input_ids"].shape[1])

    for b in range(0, len(todo), args.batch_size):
        batch = todo[b: b + args.batch_size]
        prompts, neigh, t_ret = [], [], []
        for r in batch:
            t0 = time.perf_counter()
            nbs = None
            if retriever:
                nbs = retriever.search(retriever.query_vector(r), args.k, query_lang=r["lang"],
                                       exclude_ids=[r["id"]], exclude_clusters=[r["cluster_id"]],
                                       neighbor_lang=args.neighbor_lang, mode=args.rag_mode,
                                       max_sim=args.max_sim, rng=rng)
            t_ret.append(time.perf_counter() - t0)
            neigh.append(nbs or [])
            prompts.append(apply_chat(tokenizer, build_prompt(
                r["text"], r["lang"], fmt, use_evidence, nbs, max_chars=max_chars, ref_chars=ref_chars,
                show_label=not args.hide_labels, show_text=not args.hide_text, show_signals=args.show_signals)))

        t0 = time.perf_counter()
        raws, prompt_len = generate(prompts)
        t_gen = (time.perf_counter() - t0) / len(batch)

        for r, p, raw, nbs, tr in zip(batch, prompts, raws, neigh, t_ret):
            parsed = parse_output(raw, fmt)
            row = {
                "id": r["id"], "lang": r["lang"], "host": r["host"], "section": r["section"],
                "label_gold": r["label"], "signals_gold": r["signals"],
                "label_pred": parsed["label"], "signals_pred": parsed["signals"], "evidence_pred": parsed["evidence"],
                "parse_ok": parsed["parse_ok"], "label_source": parsed["label_source"], "raw": raw,
                "neighbors": [{"id": n["id"], "label_shown": n["label"], "label_true": n["true_label"],
                               "lang": n["lang"], "sim": round(n["sim"], 4)} for n in nbs],
                "t_retrieval_s": round(tr, 4), "t_generation_s": round(t_gen, 4), "prompt_tokens_padded": prompt_len,
                "condition": condition,
            }
            if args.intervene != "none" and parsed["signals"]:
                row["interventions"] = run_interventions(args.intervene, parsed["signals"], r["lang"], p,
                                                         generate, forced_signal_prefix, parse_output, fmt)
            append_jsonl(out, row)
        print(f"[{min(b + args.batch_size, len(todo))}/{len(todo)}] {t_gen:.2f}s/article")
    print(f"✅ predictions -> {out}. Score with: python revision/stats.py --runs {out}")


def run_interventions(kind, pred_signals, lang, prompt, generate, forced_signal_prefix, parse_output, fmt):
    """Force modified signals into the assistant turn and record whether the label follows (R1-C5)."""
    variants = []
    if kind == "all_yes":
        variants.append(("all_yes", {s: "ya" for s in SIGNALS}))
    elif kind == "all_no":
        variants.append(("all_no", {s: "tidak" for s in SIGNALS}))
    else:
        for s in SIGNALS:
            mod = dict(pred_signals)
            mod[s] = flip(s, pred_signals[s])
            variants.append((f"flip_{s}", mod))
    prefixes = [forced_signal_prefix(sig, lang) for _, sig in variants]
    raws, _ = generate([prompt + pre for pre in prefixes])
    out = []
    for (name, sig), pre, raw in zip(variants, prefixes, raws):
        parsed = parse_output(pre + raw, fmt)
        out.append({"variant": name, "forced_signals": sig, "label_after": parsed["label"],
                    "rule_label": label_from_signals(sig), "raw": raw})
    return out


if __name__ == "__main__":
    main()
