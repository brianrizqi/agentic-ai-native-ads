"""
Scoring and significance testing for every prediction file (R1-C8, R1-C10, R2, minor 6). CPU only.

    python revision/stats.py --runs "revision/results/**/*.jsonl" --out-dir revision/report \
        --auto-pairs norag:rag_k5 --pair revision/results/knn/bge-m3/test__weighted_k5:revision/results/gemma3-12b/rag_k5

Run name = file path relative to --root without .jsonl. Outputs (CSV + report.md):
  metrics.csv      accuracy (Wilson 95% CI), macro-F1 (bootstrap CI), per-class P/R/F1, MCC,
                   balanced accuracy, invalid outputs, mean latency
  by_lang.csv      the same per language + Fisher exact test and Newcombe CI for the ID-EN gap
  by_group.csv     accuracy per host and per section (source-prior analysis, R2)
  signals.csv      per-signal accuracy and macro-F1 against the annotators (explanation correctness)
  mcnemar.csv      paired comparisons: discordant counts b/c, exact p, Holm and BH adjusted p,
                   accuracy difference with paired bootstrap CI, odds ratio, Cohen's g
  faithfulness.csv intervention results (label follows forced signals?)
Invalid / unparseable outputs count as errors.
"""

import argparse
import csv
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, NATIVE, NEWS, SIGNALS, SIGNAL_VALUES, read_jsonl

Z = 1.959963984540054


def wilson(k, n):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def prf(gold, pred, labels):
    out = {}
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        out[lab] = (P, R, F)
        f1s.append(F)
    return out, float(np.mean(f1s))


def mcc(gold, pred):
    tp = sum(g == NATIVE and p == NATIVE for g, p in zip(gold, pred))
    tn = sum(g == NEWS and p == NEWS for g, p in zip(gold, pred))
    fp = sum(g == NEWS and p == NATIVE for g, p in zip(gold, pred))
    fn = sum(g == NATIVE and p != NATIVE for g, p in zip(gold, pred))
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0


def bootstrap_macro_f1(gold, pred, n_boot, rng, chunk=250):
    """Macro-F1 over bootstrap resamples, vectorized (same definition as prf)."""
    codes = {lab: i for i, lab in enumerate(LABELS)}
    g = np.array([codes[x] for x in gold], dtype=np.int8)
    p = np.array([codes.get(x, -1) for x in pred], dtype=np.int8)  # invalid = -1, never a class
    out = []
    for start in range(0, n_boot, chunk):
        S = rng.integers(0, len(g), size=(min(chunk, n_boot - start), len(g)))
        G, P = g[S], p[S]
        f1s = []
        for c in range(len(LABELS)):
            tp = ((G == c) & (P == c)).sum(1)
            fp = ((G != c) & (P == c)).sum(1)
            fn = ((G == c) & (P != c)).sum(1)
            prec = np.divide(tp, tp + fp, out=np.zeros(len(tp)), where=(tp + fp) > 0)
            rec = np.divide(tp, tp + fn, out=np.zeros(len(tp)), where=(tp + fn) > 0)
            f1s.append(np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(tp)), where=(prec + rec) > 0))
        out.extend(np.mean(f1s, axis=0).tolist())
    return out


def core_metrics(rows, n_boot, rng):
    gold = [r["label_gold"] for r in rows]
    pred = [r.get("label_pred") or "invalid" for r in rows]
    n, k = len(gold), sum(g == p for g, p in zip(gold, pred))
    per, macro = prf(gold, pred, LABELS)
    lo, hi = wilson(k, n)
    boots = bootstrap_macro_f1(gold, pred, n_boot, rng) if n_boot and n else []
    rec = [per[l][1] for l in LABELS]
    return {
        "n": n, "accuracy": k / n if n else float("nan"), "acc_ci_low": lo, "acc_ci_high": hi,
        "macro_f1": macro,
        "macro_f1_ci_low": float(np.quantile(boots, .025)) if boots else "",
        "macro_f1_ci_high": float(np.quantile(boots, .975)) if boots else "",
        **{f"{p}_{lab.replace(' ', '_')}": per[lab][i] for lab in LABELS for i, p in enumerate(("precision", "recall", "f1"))},
        "mcc": mcc(gold, pred), "balanced_accuracy": float(np.mean(rec)),
        "n_invalid": sum(r.get("label_source") == "invalid" or not r.get("label_pred") for r in rows),
        "n_label_from_regex": sum(r.get("label_source") == "regex" for r in rows),
    }


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def bh(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i], reverse=True)
    adj, running = [0.0] * m, 1.0
    for rank, i in enumerate(order):
        running = min(running, ps[i] * m / (m - rank))
        adj[i] = running
    return adj


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="Files or glob patterns (quote globs)")
    ap.add_argument("--root", default="revision/results")
    ap.add_argument("--out-dir", default="revision/report")
    ap.add_argument("--pair", action="append", default=[], help="A:B run names or file paths (B minus A)")
    ap.add_argument("--auto-pairs", action="append", default=[],
                    help="cond1:cond2 -> compare <dir>/cond1 vs <dir>/cond2 for every dir that has both")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    files = sorted({p for pat in args.runs for p in glob.glob(pat, recursive=True) if p.endswith(".jsonl")})
    root = Path(args.root)

    def name_of(p):
        p = Path(p)
        try:
            return str(p.with_suffix("").relative_to(root))
        except ValueError:
            return str(p.with_suffix(""))

    runs = {}
    for f in files:
        rows = read_jsonl(f)
        if rows and "label_gold" in rows[0]:
            runs[name_of(f)] = rows
    print(f"{len(runs)} prediction files")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics, by_lang, by_group, signals, faith = [], [], [], [], []
    for name, rows in runs.items():
        m = core_metrics(rows, args.bootstrap, rng)
        tr = [r["t_retrieval_s"] for r in rows if "t_retrieval_s" in r]
        tg = [r["t_generation_s"] for r in rows if "t_generation_s" in r]
        m.update({"run": name, "mean_t_retrieval_s": float(np.mean(tr)) if tr else "",
                  "mean_t_generation_s": float(np.mean(tg)) if tg else ""})
        metrics.append(m)

        langs = sorted({r["lang"] for r in rows})
        lang_counts = {}
        for lang in langs:
            sub = [r for r in rows if r["lang"] == lang]
            lm = core_metrics(sub, args.bootstrap // 4, rng)
            lang_counts[lang] = (sum(r["label_gold"] == r.get("label_pred") for r in sub), len(sub))
            by_lang.append({"run": name, "lang": lang, **lm})
        if {"id", "en"} <= set(langs):
            from scipy.stats import fisher_exact
            (k1, n1), (k2, n2) = lang_counts["id"], lang_counts["en"]
            _, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
            lo, hi = newcombe_diff(k1, n1, k2, n2)
            by_lang.append({"run": name, "lang": "id_minus_en", "n": n1 + n2, "accuracy": k1 / n1 - k2 / n2,
                            "acc_ci_low": lo, "acc_ci_high": hi, "fisher_p": p})

        for field in ("host", "section"):
            groups = defaultdict(list)
            for r in rows:
                groups[r.get(field)].append(r)
            for g, sub in groups.items():
                k = sum(r["label_gold"] == r.get("label_pred") for r in sub)
                lo, hi = wilson(k, len(sub))
                by_group.append({"run": name, "field": field, "group": g, "n": len(sub),
                                 "n_native_gold": sum(r["label_gold"] == NATIVE for r in sub),
                                 "accuracy": k / len(sub), "acc_ci_low": lo, "acc_ci_high": hi})

        with_sig = [r for r in rows if r.get("signals_pred")]
        if with_sig:
            for s in SIGNALS:
                gold = [r["signals_gold"][s] for r in with_sig]
                pred = [r["signals_pred"][s] for r in with_sig]
                _, macro = prf(gold, pred, SIGNAL_VALUES[s])
                signals.append({"run": name, "signal": s, "n": len(with_sig), "coverage": len(with_sig) / len(rows),
                                "accuracy": float(np.mean([g == p for g, p in zip(gold, pred)])), "macro_f1": macro})
            consistent = [r for r in with_sig if r.get("label_pred")]
            from common import label_from_signals
            signals.append({"run": name, "signal": "label_matches_rule_on_own_signals", "n": len(consistent),
                            "accuracy": float(np.mean([label_from_signals(r["signals_pred"]) == r["label_pred"]
                                                       for r in consistent])) if consistent else ""})

        iv = [(r, v) for r in rows for v in r.get("interventions", [])]
        for variant in sorted({v["variant"] for _, v in iv}):
            sub = [(r, v) for r, v in iv if v["variant"] == variant]
            faith.append({"run": name, "variant": variant, "n": len(sub),
                          "label_follows_rule": float(np.mean([v["label_after"] == v["rule_label"] for _, v in sub])),
                          "label_changed": float(np.mean([v["label_after"] != r["label_pred"] for r, v in sub])),
                          "invalid_after": float(np.mean([v["label_after"] is None for _, v in sub]))})

    # ---- paired comparisons
    pairs = [tuple(p.split(":")) for p in args.pair]
    for spec in args.auto_pairs:
        a, b = spec.split(":")
        for name in runs:
            if name.endswith("/" + a) or name == a:
                other = name[: -len(a)] + b
                if other in runs:
                    pairs.append((name, other))
    mcn = []
    from scipy.stats import binomtest
    for a, b in pairs:
        a, b = name_of(a) if a.endswith(".jsonl") else a, name_of(b) if b.endswith(".jsonl") else b
        if a not in runs or b not in runs:
            print(f"⚠️  skip pair {a} vs {b}: missing run")
            continue
        A = {r["id"]: r["label_gold"] == r.get("label_pred") for r in runs[a]}
        B = {r["id"]: r["label_gold"] == r.get("label_pred") for r in runs[b]}
        ids = sorted(set(A) & set(B))
        ca, cb = np.array([A[i] for i in ids]), np.array([B[i] for i in ids])
        nb, nc = int((ca & ~cb).sum()), int((~ca & cb).sum())  # b: only A right, c: only B right
        p = binomtest(min(nb, nc), nb + nc, 0.5).pvalue if nb + nc else 1.0
        diffs = []
        for _ in range(args.bootstrap):
            s = rng.integers(0, len(ids), len(ids))
            diffs.append(cb[s].mean() - ca[s].mean())
        mcn.append({"A": a, "B": b, "n_paired": len(ids), "acc_A": ca.mean(), "acc_B": cb.mean(),
                    "diff_B_minus_A": cb.mean() - ca.mean(), "diff_ci_low": float(np.quantile(diffs, .025)),
                    "diff_ci_high": float(np.quantile(diffs, .975)), "b_only_A_correct": nb, "c_only_B_correct": nc,
                    "p_exact": p, "odds_ratio_c_over_b": (nc + .5) / (nb + .5),
                    "cohens_g": (max(nb, nc) / (nb + nc) - .5) if nb + nc else 0.0})
    if mcn:
        for key, fn in (("p_holm", holm), ("p_bh", bh)):
            for r, v in zip(mcn, fn([r["p_exact"] for r in mcn])):
                r[key] = v

    write_csv(out / "metrics.csv", metrics)
    write_csv(out / "by_lang.csv", by_lang)
    write_csv(out / "by_group.csv", by_group)
    write_csv(out / "signals.csv", signals)
    write_csv(out / "mcnemar.csv", mcn)
    write_csv(out / "faithfulness.csv", faith)

    lines = ["# Results summary", "", "| run | n | acc [95% CI] | macro-F1 [95% CI] | MCC | invalid |", "|---|---|---|---|---|---|"]
    for m in sorted(metrics, key=lambda m: -m["macro_f1"]):
        ci = f"[{m['macro_f1_ci_low']:.4f}, {m['macro_f1_ci_high']:.4f}]" if m["macro_f1_ci_low"] != "" else ""
        lines.append(f"| {m['run']} | {m['n']} | {m['accuracy']:.4f} [{m['acc_ci_low']:.4f}, {m['acc_ci_high']:.4f}] | "
                     f"{m['macro_f1']:.4f} {ci} | {m['mcc']:.4f} | {m['n_invalid']} |")
    if mcn:
        lines += ["", "| A | B | b | c | diff [95% CI] | p exact | p Holm |", "|---|---|---|---|---|---|---|"]
        for r in mcn:
            lines.append(f"| {r['A']} | {r['B']} | {r['b_only_A_correct']} | {r['c_only_B_correct']} | "
                         f"{r['diff_B_minus_A']:+.4f} [{r['diff_ci_low']:+.4f}, {r['diff_ci_high']:+.4f}] | "
                         f"{r['p_exact']:.3g} | {r['p_holm']:.3g} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\n✅ CSVs + report.md -> {out}")


if __name__ == "__main__":
    main()
