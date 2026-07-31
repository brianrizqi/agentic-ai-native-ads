"""
verify_no_leakage.py

Safety-net script: run this BEFORE any evaluation with --use-rag.
It loads the FAISS vectorstore and the evaluation dataset, and fails loudly
if any article in the evaluation set is also present in the vectorstore.

IMPORTANT: this requires the vectorstore to be rebuilt with the patched
NativeAdsVectorStore.create_from_dataset() (see note at bottom of this file)
so that each indexed document's metadata stores the ORIGINAL dataset "id"
field, not the loop position. The original create_from_dataset() stored
metadata={'id': i, ...} where i was the enumerate() index -- that index is
NOT the dataset's real article id, so it cannot be used for a real leakage
check. Fix that first (patch shown below), rebuild the vectorstore, then
run this script.

Usage:
    python verify_no_leakage.py \
        --vectorstore-dir data/vectorstore_train_only \
        --eval-dataset data/test_heldout.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectorstore-dir", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    args = parser.parse_args()

    from vector_stores.native_ads_vectorstore import NativeAdsVectorStore

    vstore = NativeAdsVectorStore(
        embedding_model=args.embedding_model,
        persist_directory=args.vectorstore_dir,
    )

    if vstore.vectorstore is None:
        print(f"FATAL: could not load vectorstore at {args.vectorstore_dir}")
        sys.exit(1)

    # Pull every document's metadata 'id' out of the FAISS docstore.
    indexed_ids = set()
    docstore = vstore.vectorstore.docstore
    for doc_id in vstore.vectorstore.index_to_docstore_id.values():
        doc = docstore.search(doc_id)
        meta_id = doc.metadata.get("id")
        indexed_ids.add(meta_id)

    with open(args.eval_dataset, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    eval_ids = {item["id"] for item in eval_data}

    overlap = indexed_ids & eval_ids

    print(f"Vectorstore documents: {len(indexed_ids)}")
    print(f"Evaluation dataset records: {len(eval_ids)}")
    print(f"Overlap: {len(overlap)}")

    if overlap:
        print(
            f"\nLEAKAGE DETECTED: {len(overlap)} article(s) in the evaluation "
            f"set are also indexed in the vectorstore. Example IDs: "
            f"{list(overlap)[:10]}"
        )
        print("Refusing to proceed. Rebuild the split and/or the vectorstore.")
        sys.exit(1)

    print("\nOK: zero overlap between vectorstore and evaluation dataset.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# REQUIRED PATCH to vector_stores/native_ads_vectorstore.py
# ---------------------------------------------------------------------------
# In create_from_dataset(), change:
#
#     for i, item in enumerate(data):
#         ...
#         doc = Document(
#             page_content=content[:1500],
#             metadata={
#                 'id': i,                      # <-- BUG: loop index, not real ID
#                 'label': label,
#                 'reasoning': reasoning,
#                 'source': 'training_dataset'
#             }
#         )
#
# to:
#
#     for i, item in enumerate(data):
#         ...
#         doc = Document(
#             page_content=content[:1500],
#             metadata={
#                 'id': item.get('id', i),      # <-- real dataset ID, for leakage checks
#                 'label': label,
#                 'reasoning': reasoning,
#                 'source': 'training_dataset'
#             }
#         )
#
# Without this patch, indexed_ids above will contain meaningless positional
# integers (0..N-1) that happen to collide with real dataset IDs by chance,
# which would make this leakage check unreliable.
