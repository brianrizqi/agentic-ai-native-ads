"""
Error diagnosis for one model's predictions, in plain text to paste back (CPU, seconds).

    python revision/diagnose.py --model-dir revision/results/gemma3-12b__reasoning_first

Compares the test partition with the withheld publisher, and Non-RAG with RAG:
  - class balance of predictions and recall per class (is the model leaning to one label?)
  - accuracy per URL section on the withheld publisher
  - agreement of each predicted signal with the annotators
  - how often the label agrees with the all-four rule applied to the model's own signals,
    and how accurate that rule-derived label would be (diagnostic only: adopting it as a
    method would have to be decided on validation data, not on these test files)
  - with RAG: accuracy by how many of the five neighbors share the gold label, and how
    often the prediction simply equals the neighbors' majority label
  - accuracy by article length
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NATIVE, NEWS, SIGNALS, label_from_signals, read_jsonl


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   n/a"


def section(title):
    print(f"\n## {title}")


def diagnose(name, rows, texts):
    n = len(rows)
    gold = [r["label_gold"] for r in rows]
    pred = [r.get("label_pred") or "invalid" for r in rows]
    ok = [g == p for g, p in zip(gold, pred)]
    print(f"\n# {name}  (n = {n}, accuracy {pct(sum(ok), n)})")

    section("Prediction balance and recall per class")
    pc, gc = Counter(pred), Counter(gold)
    print(f"gold:      native {gc[NATIVE]:5d}  news {gc[NEWS]:5d}")
    print(f"predicted: native {pc[NATIVE]:5d}  news {pc[NEWS]:5d}  invalid {pc['invalid']:4d}")
    for lab in (NATIVE, NEWS):
        idx = [i for i, g in enumerate(gold) if g == lab]
        print(f"recall {lab:12s} {pct(sum(ok[i] for i in idx), len(idx))}")
    print("confusion (gold -> predicted):")
    for (g, p), c in sorted(Counter(zip(gold, pred)).items()):
        print(f"  {g:12s} -> {p:12s} {c:5d}")

    section("Accuracy per URL section (sections with >= 20 articles)")
    by = defaultdict(list)
    for r, o, p in zip(rows, ok, pred):
        by[r.get("section") or r.get("host")].append((o, p == NATIVE, r["label_gold"] == NATIVE))
    for sec, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(v) >= 20:
            print(f"  {str(sec)[:24]:24s} n={len(v):4d}  acc {pct(sum(a for a, _, _ in v), len(v))}  "
                  f"gold native {pct(sum(g for _, _, g in v), len(v))}  predicted native {pct(sum(p for _, p, _ in v), len(v))}")

    with_sig = [r for r in rows if r.get("signals_pred")]
    section(f"Signals vs annotators ({len(with_sig)} of {n} outputs carry signals)")
    for s in SIGNALS:
        agree = sum(r["signals_pred"][s] == r["signals_gold"][s] for r in with_sig)
        gpos = sum(r["signals_gold"][s] == "ya" for r in with_sig)
        ppos = sum(r["signals_pred"][s] == "ya" for r in with_sig)
        print(f"  {s:20s} agreement {pct(agree, len(with_sig))}   gold 'ya' {pct(gpos, len(with_sig))}   predicted 'ya' {pct(ppos, len(with_sig))}")
    if with_sig:
        rule = [label_from_signals(r["signals_pred"]) for r in with_sig]
        cons = sum(rl == r.get("label_pred") for rl, r in zip(rule, with_sig))
        racc = sum(rl == r["label_gold"] for rl, r in zip(rule, with_sig))
        macc = sum(r.get("label_pred") == r["label_gold"] for r in with_sig)
        print(f"  label = all-four rule on own signals: {pct(cons, len(with_sig))}")
        print(f"  accuracy of model label {pct(macc, len(with_sig))}  vs  rule-derived label {pct(racc, len(with_sig))}  (diagnostic only)")

    nb = [r for r in rows if r.get("neighbors")]
    if nb:
        section("Retrieval: accuracy by neighbors sharing the gold label")
        groups = defaultdict(list)
        copies = 0
        for r in nb:
            labels = [x["label_shown"] for x in r["neighbors"]]
            same = sum(l == r["label_gold"] for l in labels)
            groups[same].append(r.get("label_pred") == r["label_gold"])
            maj = Counter(labels).most_common(1)[0][0]
            copies += r.get("label_pred") == maj
        for k in sorted(groups):
            v = groups[k]
            print(f"  {k}/5 neighbors agree with gold: n={len(v):4d}  acc {pct(sum(v), len(v))}")
        print(f"  prediction equals neighbors' majority label: {pct(copies, len(nb))}")
        sims = sorted(max(x["sim"] for x in r["neighbors"]) for r in nb)
        print(f"  top-1 neighbor similarity: median {sims[len(sims) // 2]:.3f}, 10th pct {sims[len(sims) // 10]:.3f}")

    if texts:
        section("Accuracy by article length (characters after cleaning)")
        bands = [(0, 1000), (1000, 2000), (2000, 3500), (3500, 10 ** 9)]
        for lo, hi in bands:
            v = [o for r, o in zip(rows, ok) if lo <= len(texts.get(r["id"], "")) < hi]
            print(f"  {lo:5d}-{'max' if hi > 10 ** 8 else hi:<5}  n={len(v):4d}  acc {pct(sum(v), len(v))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="revision/results/<model>__<format>")
    ap.add_argument("--split-dir", default="revision/splits/main")
    args = ap.parse_args()
    d = Path(args.model_dir)
    texts = {}
    for f in ("test.jsonl", "test_xsource.jsonl"):
        p = Path(args.split_dir) / f
        if p.exists():
            texts.update({r["id"]: r["text"] for r in read_jsonl(p)})
    for cond in ("test__norag", "test__rag_k5", "test_xsource__norag", "test_xsource__rag_k5"):
        f = d / f"{cond}.jsonl"
        if f.exists():
            diagnose(cond, read_jsonl(f), texts)
    a, b = d / "test_xsource__norag.jsonl", d / "test_xsource__rag_k5.jsonl"
    if a.exists() and b.exists():
        A = {r["id"]: r["label_gold"] == r.get("label_pred") for r in read_jsonl(a)}
        B = {r["id"]: r["label_gold"] == r.get("label_pred") for r in read_jsonl(b)}
        ids = set(A) & set(B)
        print(f"\n# Withheld publisher, Non-RAG vs RAG: only RAG right {sum(B[i] and not A[i] for i in ids)}, "
              f"only Non-RAG right {sum(A[i] and not B[i] for i in ids)}")


if __name__ == "__main__":
    main()
