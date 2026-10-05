"""
Shortcut probe (R2: category/source priors; R1 Q1, Q8). CPU only, about a minute.

    python revision/shortcut_probe.py --split-dir revision/splits/main --out revision/results/shortcut_probe.json

Answers three questions with simple, transparent models:
  1. How much of the label is predictable from the source alone (host / section majority label)?
  2. How well does a bag-of-words classifier do with and without the outlet/wire datelines?
  3. Does that classifier transfer to a publisher it never saw (test_xsource)?
High scores in (1) or a large raw-vs-masked gap in (2) are evidence of a source shortcut
and must be reported alongside the LLM results.
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_jsonl, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    d = Path(args.split_dir)
    train = read_jsonl(d / "train.jsonl")
    tests = {s: read_jsonl(d / f"{s}.jsonl") for s in ("test", "test_xsource") if (d / f"{s}.jsonl").exists()}
    result = {}

    # 1. source-only baselines
    for field in ("host", "section"):
        maj = {k: Counter(v).most_common(1)[0][0] for k, v in _group(train, field).items()}
        overall = Counter(r["label"] for r in train).most_common(1)[0][0]
        for name, rows in tests.items():
            pred = [maj.get(r[field], overall) for r in rows]
            gold = [r["label"] for r in rows]
            result[f"{name}__{field}_majority"] = _scores(gold, pred, rows)
    result["label_by_host_train"] = {h: dict(Counter(v)) for h, v in _group(train, "host").items()}

    # 2-3. TF-IDF + logistic regression, raw text vs cue-masked text
    for variant in ("text_raw", "text"):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
        X = vec.fit_transform([r[variant][:3000] for r in train])
        clf = LogisticRegression(max_iter=2000, C=4.0).fit(X, [r["label"] for r in train])
        for name, rows in tests.items():
            pred = clf.predict(vec.transform([r[variant][:3000] for r in rows]))
            result[f"{name}__tfidf_lr__{'raw' if variant == 'text_raw' else 'masked'}"] = \
                _scores([r["label"] for r in rows], list(pred), rows)
        if variant == "text_raw":
            names = vec.get_feature_names_out()
            coef = clf.coef_[0]
            top = coef.argsort()
            result["tfidf_raw_top_features"] = {clf.classes_[0]: [names[i] for i in top[:25]],
                                                clf.classes_[1]: [names[i] for i in top[-25:][::-1]]}
    write_json(args.out, result)
    for k, v in result.items():
        if isinstance(v, dict) and "accuracy" in v:
            print(f"{k:45s} acc={v['accuracy']:.4f} macroF1={v['macro_f1']:.4f}  by_lang={v['by_lang']}")
    print(f"✅ -> {args.out}")


def _group(rows, field):
    g = defaultdict(list)
    for r in rows:
        g[r[field]].append(r["label"])
    return g


def _scores(gold, pred, rows):
    from sklearn.metrics import accuracy_score, f1_score
    by_lang = {}
    for lang in sorted({r["lang"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["lang"] == lang]
        by_lang[lang] = round(accuracy_score([gold[i] for i in idx], [pred[i] for i in idx]), 4)
    return {"n": len(gold), "accuracy": accuracy_score(gold, pred),
            "macro_f1": f1_score(gold, pred, average="macro"), "by_lang": by_lang}


if __name__ == "__main__":
    main()
