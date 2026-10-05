"""
Retrieval-only baselines and retrieval diagnostics (R1-C2, R1-C3, R1-C4, R2). CPU is enough.

    python revision/knn_baseline.py --index-dir revision/index/bge-m3 \
        --split-dir revision/splits/main --out-dir revision/results/knn/bge-m3

Writes, for test.jsonl and test_xsource.jsonl:
  <split>__1nn.jsonl, <split>__majority_k{K}.jsonl, <split>__weighted_k{K}.jsonl   (same schema as infer.py)
  <split>__weighted_k5_lang-{same,cross}.jsonl     cross-lingual retrieval
  <split>__weighted_k5_maxsim0.90.jsonl            neighbors above 0.90 similarity removed
  diagnostics.json                                 similarity distribution, near-duplicates, neighbor-label purity
Run once per encoder index to compare encoders (MiniLM vs bge-m3 vs e5).
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NATIVE, NEWS, read_json, read_jsonl, write_json, write_jsonl


def vote(labels, sims, weighted):
    score = Counter()
    for lab, s in zip(labels, sims):
        score[lab] += float(s) if weighted else 1.0
    if score[NATIVE] == score[NEWS]:
        return labels[0]  # tie -> nearest neighbor
    return NATIVE if score[NATIVE] > score[NEWS] else NEWS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ks", default="1,3,5,7,10")
    args = ap.parse_args()

    idx = Path(args.index_dir)
    info = read_json(idx / "info.json")
    train_emb = np.load(idx / "train_emb.npy")
    meta = read_jsonl(idx / "train_meta.jsonl")
    tr_lab = np.array([m["label"] for m in meta])
    tr_lang = np.array([m["lang"] for m in meta])
    ks = [int(k) for k in args.ks.split(",")]
    out = Path(args.out_dir)
    diagnostics = {"encoder": info["model"]}

    for split in ("test", "test_xsource"):
        f = Path(args.split_dir) / f"{split}.jsonl"
        if not f.exists():
            continue
        rows = read_jsonl(f)
        ids = read_json(idx / f"{split}_ids.json")
        assert ids == [r["id"] for r in rows], f"{split} embeddings out of order; rebuild the index"
        q = np.load(idx / f"{split}_emb.npy")
        sims = q @ train_emb.T
        order = np.argsort(-sims, axis=1)[:, :max(ks)]

        def write(name, preds, extra=None):
            write_jsonl(out / f"{split}__{name}.jsonl", (
                {"id": r["id"], "lang": r["lang"], "host": r["host"], "section": r["section"],
                 "label_gold": r["label"], "signals_gold": r["signals"], "label_pred": p,
                 "parse_ok": True, "label_source": "knn", "condition": {"method": name, "encoder": info["model"],
                                                                       **(extra or {})}}
                for r, p in zip(rows, preds)))

        write("1nn", [tr_lab[o[0]] for o in order])
        for k in ks:
            write(f"majority_k{k}", [vote(tr_lab[o[:k]], sims[i, o[:k]], False) for i, o in enumerate(order)])
            write(f"weighted_k{k}", [vote(tr_lab[o[:k]], sims[i, o[:k]], True) for i, o in enumerate(order)])

        # cross-lingual retrieval: restrict neighbors by language relative to the query
        for mode in ("same", "cross"):
            preds = []
            for i, r in enumerate(rows):
                mask = (tr_lang == r["lang"]) if mode == "same" else (tr_lang != r["lang"])
                cand = np.flatnonzero(mask)
                top = cand[np.argsort(-sims[i, cand])[:5]]
                preds.append(vote(tr_lab[top], sims[i, top], True))
            write(f"weighted_k5_lang-{mode}", preds, {"neighbor_lang": mode})

        # near-duplicate control: drop neighbors above a similarity ceiling
        for ceiling in (0.95, 0.90, 0.85):
            preds = []
            for i in range(len(rows)):
                cand = np.flatnonzero(sims[i] <= ceiling)
                top = cand[np.argsort(-sims[i, cand])[:5]]
                preds.append(vote(tr_lab[top], sims[i, top], True))
            write(f"weighted_k5_maxsim{ceiling:.2f}", preds, {"max_sim": ceiling})

        top1 = sims[np.arange(len(rows)), order[:, 0]]
        gold = np.array([r["label"] for r in rows])
        purity5 = (tr_lab[order[:, :5]] == gold[:, None]).mean(axis=1)
        langs = np.array([r["lang"] for r in rows])
        nb_lang5 = tr_lang[order[:, :5]]
        diagnostics[split] = {
            "n": len(rows),
            "top1_similarity_quantiles": {str(p): float(np.quantile(top1, p)) for p in (0, .05, .25, .5, .75, .95, 1)},
            "n_queries_with_neighbor_sim_ge": {str(t): int((top1 >= t).sum()) for t in (0.85, 0.90, 0.95, 0.98)},
            "neighbor_label_purity_k5": {"mean": float(purity5.mean()),
                                         "distribution": {f"{int(round(v * 5))}/5": int((np.round(purity5 * 5) == round(v * 5)).sum())
                                                          for v in np.unique(purity5)},
                                         "by_lang": {l: float(purity5[langs == l].mean()) for l in np.unique(langs)}},
            "neighbor_same_language_share_k5": {l: float((nb_lang5[langs == l] == l).mean()) for l in np.unique(langs)},
            "top1_similarity_mean_by_correct_1nn": {
                "correct": float(top1[tr_lab[order[:, 0]] == gold].mean()),
                "wrong": float(top1[tr_lab[order[:, 0]] != gold].mean()) if (tr_lab[order[:, 0]] != gold).any() else None},
        }
    write_json(out / "diagnostics.json", diagnostics)
    print(f"✅ kNN predictions + diagnostics -> {out}")


if __name__ == "__main__":
    main()
