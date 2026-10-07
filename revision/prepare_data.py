"""
Step 1: deduplicate the corpus and freeze a leakage-free split (R1-C1, R1-C2, R2, R3).

    python revision/prepare_data.py --out-dir revision/splits/main

What it does, in order:
  1. Loads native_ads_dataset.xlsx (url, content, four signal annotations, label, lang)
     and checks it row-aligns with data/llm_dataset_12k_refined_full.json.
  2. Strips outlet/wire datelines ("(GLOBE NEWSWIRE) —", "KOMPAS.com -", ...) so models
     cannot classify by source formatting (--no-mask-cues to disable).
  3. Clusters duplicates: exact normalized-text hash + MinHash word-shingle Jaccard,
     plus optional multilingual semantic similarity (catches translations/rewrites).
  4. Assigns whole clusters to train/val/test, stratified by (lang, label), so no member
     of a duplicate cluster crosses partitions.
  5. Optionally moves every article from --holdout-hosts into test_xsource.jsonl, which
     no model ever trains on: a cross-publisher generalization test.

The test files are written once and must never be edited, used for prompt design,
hyperparameter selection or qualitative example selection.
"""

import argparse
import json
import random
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO, SIGNAL_COLUMNS, SIGNALS, env_versions, mask_source_cues, norm_label,
                    norm_signal, normalize_for_hash, parse_url, sha256, write_json, write_jsonl)


# ---------------------------------------------------------------- union-find

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# ---------------------------------------------------------------- loading

def load_records(xlsx_path, json_path, mask_cues, min_chars):
    import pandas as pd
    df = pd.read_excel(xlsx_path)
    records = []
    for _, row in df.iterrows():
        label = norm_label(row["label"])
        signals = {s: norm_signal(s, row[SIGNAL_COLUMNS[s]]) for s in SIGNALS}
        raw = str(row["content"] or "")
        rec = {
            "id": str(row["id"]),
            "lang": str(row["lang"]).strip().lower(),
            "label": label,
            "signals": signals,
            "url": str(row["url"]),
            **parse_url(row["url"]),
            "text_raw": raw,
            "text": mask_source_cues(raw) if mask_cues else raw,
        }
        records.append(rec)

    if json_path and Path(json_path).exists():
        with open(json_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        assert len(js) == len(records), f"row count differs: xlsx={len(records)} json={len(js)}"
        for r, j in zip(records, js):
            assert r["text_raw"].strip()[:200] == j["input"].strip()[:200], f"row misaligned at id {r['id']}"
            assert r["label"] == norm_label(json.loads(j["output"])["label"]), f"label differs at id {r['id']}"
        print(f"✓ xlsx row-aligns with {json_path} ({len(js)} rows, content and label)")

    bad = [r["id"] for r in records if r["label"] is None or any(v is None for v in r["signals"].values())]
    if bad:
        print(f"⚠️  {len(bad)} records with missing label/signal, dropped: {bad[:10]}")
    short = [r["id"] for r in records if len(r["text"].strip()) < min_chars]
    if short:
        print(f"⚠️  {len(short)} records shorter than {min_chars} chars (scraping residue), dropped: {short[:10]}")
    drop = set(bad) | set(short)
    records = [r for r in records if r["id"] not in drop]
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate ids in source"
    return records


# ---------------------------------------------------------------- dedup

def shingles(text, n=5):
    toks = normalize_for_hash(text).split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def minhash_pairs(texts, num_perm=128, bands=32, threshold=0.7, seed=1):
    """Candidate pairs via MinHash LSH, verified with exact Jaccard."""
    p = np.uint64(2147483647)
    rng = np.random.RandomState(seed)
    a = rng.randint(1, 2147483647, size=num_perm).astype(np.uint64)
    b = rng.randint(0, 2147483647, size=num_perm).astype(np.uint64)
    sh_sets = [shingles(t) for t in texts]
    sigs = np.full((len(texts), num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, s in enumerate(sh_sets):
        if not s:
            continue
        x = np.array([zlib.crc32(g.encode("utf-8")) for g in s], dtype=np.uint64) % p
        sigs[i] = ((np.outer(x, a) + b) % p).min(axis=0)
    rows = num_perm // bands
    buckets = defaultdict(list)
    for i in range(len(texts)):
        for bnd in range(bands):
            buckets[(bnd, sigs[i, bnd * rows:(bnd + 1) * rows].tobytes())].append(i)
    cand = set()
    for members in buckets.values():
        if 1 < len(members) <= 200:
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    cand.add((members[x], members[y]))
    pairs = []
    for i, j in cand:
        si, sj = sh_sets[i], sh_sets[j]
        if si and sj:
            jac = len(si & sj) / len(si | sj)
            if jac >= threshold:
                pairs.append((i, j, jac))
    return pairs


def semantic_pairs(texts, model_name, threshold, cache=None, batch_size=32, chunk=2048):
    if cache and Path(cache).exists():
        emb = np.load(cache)
        print(f"   loaded cached embeddings {cache} {emb.shape}")
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        emb = model.encode([t[:2000] for t in texts], batch_size=batch_size,
                           normalize_embeddings=True, show_progress_bar=True).astype("float32")
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, emb)
    pairs = []
    for s in range(0, len(emb), chunk):
        sims = emb[s:s + chunk] @ emb.T
        ii, jj = np.where(sims >= threshold)
        for i, j in zip(ii + s, jj):
            if i < j:
                pairs.append((int(i), int(j), float(sims[i - s, j])))
    return pairs


# ---------------------------------------------------------------- split

def split_clusters(records, clusters, fracs, seed, test_frac_by_lang):
    """Greedy cluster assignment per (lang, majority label) stratum."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for cid, members in clusters.items():
        lang = Counter(records[i]["lang"] for i in members).most_common(1)[0][0]
        lab = Counter(records[i]["label"] for i in members).most_common(1)[0][0]
        strata[(lang, lab)].append(cid)
    assignment = {}
    for key in sorted(strata):
        cids = sorted(strata[key])
        rng.shuffle(cids)
        total = sum(len(clusters[c]) for c in cids)
        test_frac = test_frac_by_lang.get(key[0], fracs["test"])
        quota = {"test": test_frac * total, "val": fracs["val"] * total}
        filled = {"test": 0, "val": 0}
        for c in cids:
            n = len(clusters[c])
            if filled["test"] + n <= quota["test"] + 0.5:
                assignment[c], filled["test"] = "test", filled["test"] + n
            elif filled["val"] + n <= quota["val"] + 0.5:
                assignment[c], filled["val"] = "val", filled["val"] + n
            else:
                assignment[c] = "train"
    return assignment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", default=str(REPO / "native_ads_dataset.xlsx"))
    ap.add_argument("--json", default=str(REPO / "data" / "llm_dataset_12k_refined_full.json"),
                    help="Only used to verify row alignment with the dataset used in the first submission")
    ap.add_argument("--out-dir", default=str(REPO / "revision" / "splits" / "main"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--test-frac-en", type=float, default=0.25,
                    help="Larger English test share so per-language results are not underpowered (R1-C8, R2)")
    ap.add_argument("--holdout-hosts", default="cnnindonesia.com",
                    help="Comma-separated hosts excluded from training entirely -> test_xsource.jsonl. "
                         "cnnindonesia.com is label-balanced (1163/1221), so it tests within-outlet discrimination. "
                         "Pass '' to disable.")
    ap.add_argument("--min-chars", type=int, default=100, help="Drop scraping residue shorter than this")
    ap.add_argument("--no-mask-cues", action="store_true", help="Keep outlet/wire datelines in the text")
    ap.add_argument("--jaccard", type=float, default=0.7, help="MinHash near-duplicate threshold")
    ap.add_argument("--semantic-model", default="BAAI/bge-m3",
                    help="Multilingual encoder for semantic/cross-language duplicates; '' to skip")
    ap.add_argument("--semantic-threshold", type=float, default=0.95)
    ap.add_argument("--emb-cache", default=None, help="Cache file for the semantic-dedup embeddings (.npy)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(args.xlsx, args.json, mask_cues=not args.no_mask_cues, min_chars=args.min_chars)
    n = len(records)
    print(f"Loaded {n} records")

    # ---- dedup
    uf = UnionFind(n)
    dup_edges = Counter()
    by_hash = defaultdict(list)
    for i, r in enumerate(records):
        r["text_sha256"] = sha256(normalize_for_hash(r["text_raw"]))
        by_hash[r["text_sha256"]].append(i)
    for members in by_hash.values():
        for j in members[1:]:
            uf.union(members[0], j)
            dup_edges["exact"] += 1
    print(f"exact/normalized duplicates: {dup_edges['exact']} links")

    print("MinHash near-duplicate search ...")
    for i, j, _ in minhash_pairs([r["text_raw"] for r in records], threshold=args.jaccard):
        uf.union(i, j)
        dup_edges["minhash"] += 1
    print(f"MinHash (Jaccard >= {args.jaccard}): {dup_edges['minhash']} links")

    sem_pairs = []
    if args.semantic_model:
        print(f"Semantic duplicate search with {args.semantic_model} (cos >= {args.semantic_threshold}) ...")
        sem_pairs = semantic_pairs([r["text"] for r in records], args.semantic_model,
                                   args.semantic_threshold, cache=args.emb_cache)
        for i, j, _ in sem_pairs:
            uf.union(i, j)
            dup_edges["semantic"] += 1
            if records[i]["lang"] != records[j]["lang"]:
                dup_edges["semantic_cross_language"] += 1
        print(f"semantic: {dup_edges['semantic']} links ({dup_edges['semantic_cross_language']} cross-language)")

    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)
    cluster_ids = {root: f"c{k:05d}" for k, root in enumerate(sorted(clusters))}
    for root, members in clusters.items():
        for i in members:
            records[i]["cluster_id"] = cluster_ids[root]
    multi = [m for m in clusters.values() if len(m) > 1]
    conflicting = [m for m in multi if len({records[i]["label"] for i in m}) > 1]
    print(f"clusters: {len(clusters)} ({len(multi)} with >1 member, {sum(map(len, multi))} articles; "
          f"{len(conflicting)} with conflicting labels)")

    # ---- holdout hosts (cross-publisher test)
    holdout = {h.strip() for h in args.holdout_hosts.split(",") if h.strip()}
    holdout_roots = {uf.find(i) for i, r in enumerate(records) if r["host"] in holdout}
    xsource, excluded = [], []
    remaining = {}
    for root, members in clusters.items():
        if root in holdout_roots:
            for i in members:
                (xsource if records[i]["host"] in holdout else excluded).append(records[i])
        else:
            remaining[root] = members

    # ---- split
    fracs = {"val": args.val_frac, "test": args.test_frac}
    assignment = split_clusters(records, remaining, fracs, args.seed, {"en": args.test_frac_en})
    parts = defaultdict(list)
    for root, members in remaining.items():
        for i in members:
            records[i]["split"] = assignment[root]
            parts[assignment[root]].append(records[i])
    for r in xsource:
        r["split"] = "test_xsource"
    for r in excluded:
        r["split"] = "excluded"

    # Hard leakage checks
    split_of_cluster = defaultdict(set)
    for r in records:
        split_of_cluster[r["cluster_id"]].add(r["split"])
    crossing = [c for c, s in split_of_cluster.items() if len(s - {"excluded"}) > 1]
    assert not crossing, f"FATAL: clusters crossing partitions: {crossing[:5]}"
    train_ids = {r["id"] for r in parts["train"]}
    for name in ("val", "test"):
        assert not train_ids & {r["id"] for r in parts[name]}, f"FATAL: train/{name} overlap"
    assert not train_ids & {r["id"] for r in xsource}, "FATAL: train/xsource overlap"
    assert not any(r["host"] in holdout for r in parts["train"] + parts["val"]), "FATAL: holdout host in train/val"

    rng = random.Random(args.seed)
    files = {}
    for name in ("train", "val", "test"):
        rows = sorted(parts[name], key=lambda r: r["id"])
        rng.shuffle(rows)
        files[name] = out / f"{name}.jsonl"
        write_jsonl(files[name], rows)
    if xsource:
        files["test_xsource"] = out / "test_xsource.jsonl"
        write_jsonl(files["test_xsource"], sorted(xsource, key=lambda r: r["id"]))
    if excluded:
        write_jsonl(out / "excluded.jsonl", excluded)

    # ---- manifest (R1-C12)
    import csv
    with open(out / "ids_hashes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Source URLs are deliberately left out: article URLs and texts are not redistributed.
        w.writerow(["id", "split", "cluster_id", "text_sha256", "lang", "label", "host", "section"])
        for r in sorted(records, key=lambda r: int(r["id"]) if r["id"].isdigit() else r["id"]):
            w.writerow([r["id"], r["split"], r["cluster_id"], r["text_sha256"], r["lang"], r["label"], r["host"], r["section"]])

    def dist(rows):
        c = Counter((r["lang"], r["label"]) for r in rows)
        return {f"{k[0]}|{k[1]}": v for k, v in sorted(c.items())}

    import hashlib
    manifest = {
        "args": vars(args),
        "n_records": n,
        "dedup_links": dict(dup_edges),
        "n_clusters": len(clusters),
        "n_multi_member_clusters": len(multi),
        "n_articles_in_multi_member_clusters": sum(map(len, multi)),
        "n_conflicting_label_clusters": len(conflicting),
        "counts": {name: {"n": len(parts[name]), "by_lang_label": dist(parts[name])} for name in ("train", "val", "test")},
        "test_xsource": {"hosts": sorted(holdout), "n": len(xsource), "by_lang_label": dist(xsource)},
        "excluded_same_cluster_as_holdout": len(excluded),
        "file_sha256": {k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in files.items()},
        "versions": env_versions(),
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("counts", "test_xsource", "dedup_links")}, indent=2))
    print(f"\n✅ Split frozen in {out}. Commit manifest.json + ids_hashes.csv; never edit test*.jsonl.")


if __name__ == "__main__":
    main()
