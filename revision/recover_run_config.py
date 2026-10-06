"""
Rebuild run_config.json for an adapter whose training finished but whose final
bookkeeping step crashed (the adapter weights were already saved).

    python revision/recover_run_config.py --out revision/models/gemma3-4b__reasoning_first --model gemma3-4b

Training metrics and the best checkpoint come from the trainer_state.json that the
Trainer wrote into the last checkpoint. The file is marked "recovered": true.
Arguments must match the original finetune.py call; defaults match run_all.sh.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import env_versions, read_jsonl, write_json
from finetune import MODELS, SHARED, split_hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--format", default="reasoning_first")
    ap.add_argument("--split-dir", default="revision/splits/main")
    ap.add_argument("--index-dir", default="revision/index/bge-m3")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rag-dropout", type=float, default=0.5)
    ap.add_argument("--max-val", type=int, default=400)
    args = ap.parse_args()

    out = Path(args.out)
    assert (out / "adapter_config.json").exists(), f"no saved adapter in {out}"
    if (out / "run_config.json").exists():
        sys.exit(f"{out}/run_config.json already exists, nothing to do")
    ckpts = sorted((out / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    assert ckpts, f"no checkpoints in {out}/checkpoints"
    state = json.load(open(ckpts[-1] / "trainer_state.json"))
    train_metrics = next((h for h in reversed(state["log_history"]) if "train_runtime" in h), {})

    train = read_jsonl(Path(args.split_dir) / "train.jsonl")
    val = read_jsonl(Path(args.split_dir) / "val.jsonl")
    rng = random.Random(SHARED["seed"])  # same draw as build_examples in finetune.py
    n_rag = sum(rng.random() >= args.rag_dropout for _ in train)

    write_json(out / "run_config.json", {
        "recovered": True, "recovered_from": str(ckpts[-1]),
        "model_key": args.model, "base_model": MODELS.get(args.model, args.model), "format": args.format,
        "use_evidence": False, "rag": True, "index_dir": args.index_dir, "k": args.k,
        "rag_dropout": args.rag_dropout, "max_chars": 1500, "neighbor_chars": 400,
        "hyperparameters": dict(SHARED), "overrides": {}, "effective_batch": SHARED["per_device_batch"] * SHARED["grad_accum"],
        "n_train": len(train), "n_val": min(args.max_val, len(val)), "n_train_with_references": n_rag,
        "loss": "answer tokens only", "split_dir": args.split_dir, "split_file_sha256": split_hashes(args.split_dir),
        "train_metrics": train_metrics, "best_checkpoint": state.get("best_model_checkpoint"),
        "best_eval_loss": state.get("best_metric"), "log_history": state["log_history"], "versions": env_versions(),
    })
    print(f"✅ wrote {out}/run_config.json (best eval_loss {state.get('best_metric')})")


if __name__ == "__main__":
    main()
