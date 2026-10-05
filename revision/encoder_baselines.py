"""
Encoder baselines retrained on the SAME leakage-free split (R1-C9). One GPU, ~20-40 min each.

    python revision/encoder_baselines.py --model indobenchmark/indobert-base-p1 \
        --split-dir revision/splits/main --out-dir revision/results/encoders/indobert
    # also: xlm-roberta-base, bert-base-multilingual-cased (and xlm-roberta-large if time allows)

Model selection uses val.jsonl only (macro-F1). Predictions for test.jsonl and
test_xsource.jsonl are written in the infer.py schema so stats.py scores them identically.
SENADA must be rerun with its own released code on train.jsonl / test.jsonl: see RUNBOOK.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, env_versions, read_jsonl, write_json, write_jsonl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, set_seed)

    set_seed(args.seed)
    d, out = Path(args.split_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lab2id = {l: i for i, l in enumerate(LABELS)}
    tok = AutoTokenizer.from_pretrained(args.model)

    def ds(rows):
        enc = tok([r["text"] for r in rows], truncation=True, max_length=args.max_length)
        return Dataset.from_dict({**enc, "labels": [lab2id[r["label"]] for r in rows]})

    train, val = read_jsonl(d / "train.jsonl"), read_jsonl(d / "val.jsonl")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    def metrics(p):
        return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(out / "ckpt"), num_train_epochs=args.epochs, learning_rate=args.lr,
                               per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=64,
                               eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
                               load_best_model_at_end=True, metric_for_best_model="macro_f1", seed=args.seed,
                               bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                               report_to="none", warmup_ratio=0.06, weight_decay=0.01),
        train_dataset=ds(train), eval_dataset=ds(val), data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics)
    trainer.train()

    for split in ("test", "test_xsource"):
        f = d / f"{split}.jsonl"
        if not f.exists():
            continue
        rows = read_jsonl(f)
        logits = trainer.predict(ds(rows)).predictions
        prob = torch.softmax(torch.tensor(logits), -1).numpy()
        write_jsonl(out / f"{split}.jsonl", (
            {"id": r["id"], "lang": r["lang"], "host": r["host"], "section": r["section"],
             "label_gold": r["label"], "signals_gold": r["signals"], "label_pred": LABELS[int(np.argmax(p))],
             "prob_native": float(p[lab2id["native ads"]]), "parse_ok": True, "label_source": "classifier",
             "condition": {"model": args.model, "method": "encoder_finetune"}}
            for r, p in zip(rows, prob)))
    write_json(out / "run_config.json", {**vars(args), "best_checkpoint": trainer.state.best_model_checkpoint,
                                         "best_val_macro_f1": trainer.state.best_metric, "versions": env_versions()})
    print(f"✅ -> {out}")


if __name__ == "__main__":
    main()
