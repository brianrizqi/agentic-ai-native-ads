"""
Step 2: embed the TRAIN split into the retrieval index and cache query embeddings
for val/test/test_xsource (R1-C4, R3: multilingual encoder).

    python revision/build_index.py --split-dir revision/splits/main \
        --model BAAI/bge-m3 --out-dir revision/index/bge-m3

Build one index per encoder you want to compare, e.g. also
sentence-transformers/all-MiniLM-L6-v2 (first-submission encoder) and
intfloat/multilingual-e5-large (--prefix "query: ").
Only train.jsonl enters the index; the other splits are embedded as queries.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import env_versions, read_jsonl, write_json, write_jsonl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="", help='Text prefix, e.g. "query: " for multilingual-e5')
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device=args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.split_dir)

    def embed(rows):
        return model.encode([args.prefix + r["text"][: args.max_chars] for r in rows], batch_size=args.batch_size,
                            normalize_embeddings=True, show_progress_bar=True).astype("float32")

    train = read_jsonl(split_dir / "train.jsonl")
    np.save(out / "train_emb.npy", embed(train))
    write_jsonl(out / "train_meta.jsonl", ({k: r[k] for k in ("id", "label", "lang", "signals", "cluster_id",
                                                                "host", "text")} for r in train))
    for split in ("val", "test", "test_xsource"):
        f = split_dir / f"{split}.jsonl"
        if f.exists():
            rows = read_jsonl(f)
            np.save(out / f"{split}_emb.npy", embed(rows))
            write_json(out / f"{split}_ids.json", [r["id"] for r in rows])
    write_json(out / "info.json", {"model": args.model, "prefix": args.prefix, "max_chars": args.max_chars,
                                   "split_dir": str(split_dir), "n_train": len(train),
                                   "split_manifest_sha256": read_manifest_hash(split_dir),
                                   "versions": env_versions()})
    print(f"✅ index with {len(train)} train articles -> {out}")


def read_manifest_hash(split_dir):
    import hashlib
    p = Path(split_dir) / "manifest.json"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


if __name__ == "__main__":
    main()
