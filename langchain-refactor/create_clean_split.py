"""
create_clean_split.py

Creates a stratified train/test split by (lang, label), with a fixed seed,
from the FULL 14,956-article dataset. The train split is what the RAG
vectorstore must be built from. The test split is the ONLY file that may be
used for evaluation. The two are guaranteed to be disjoint at the ID level.

Usage:
    python create_clean_split.py \
        --input data/llm_dataset_12k_refined_full.json \
        --train-output data/train_split.json \
        --test-output data/test_heldout.json \
        --test-size 0.15 \
        --seed 42
"""

import argparse
import json
import random
from collections import defaultdict


def get_label(item):
    try:
        return json.loads(item["output"]).get("label", "unknown")
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--test-output", required=True)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} total records from {args.input}")

    # Stratify by (lang, label) so both partitions preserve the same
    # language and class balance as the full corpus.
    groups = defaultdict(list)
    for item in data:
        lang = item.get("lang", "id")
        label = get_label(item)
        groups[(lang, label)].append(item)

    rng = random.Random(args.seed)

    train_data, test_data = [], []
    print("\nStratified split by (lang, label):")
    for key, items in sorted(groups.items()):
        items = items[:]  # copy
        rng.shuffle(items)
        n_test = max(1, round(len(items) * args.test_size))
        test_part = items[:n_test]
        train_part = items[n_test:]
        train_data.extend(train_part)
        test_data.extend(test_part)
        print(f"  {key}: total={len(items)}  train={len(train_part)}  test={len(test_part)}")

    rng.shuffle(train_data)
    rng.shuffle(test_data)

    # Hard safety check: assert disjoint IDs before writing anything out.
    train_ids = {item["id"] for item in train_data}
    test_ids = {item["id"] for item in test_data}
    overlap = train_ids & test_ids
    assert not overlap, f"FATAL: {len(overlap)} IDs appear in both splits: {list(overlap)[:10]}"

    with open(args.train_output, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(args.test_output, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"\nTrain split: {len(train_data)} records -> {args.train_output}")
    print(f"Test split (held-out): {len(test_data)} records -> {args.test_output}")
    print("Verified: zero ID overlap between train and test splits.")


if __name__ == "__main__":
    main()
