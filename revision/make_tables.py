"""
Turn the CSVs written by stats.py into LaTeX tables for the manuscript (CPU, seconds).

    python revision/make_tables.py --report-dir revision/report --out-dir revision/report/latex

Writes one .tex file per table (booktabs, \\tblwidth as in rana_ieee_access.tex), so no
number is copied into the paper by hand. A table whose inputs are missing is skipped
with a message. Accuracies are percentages with two decimals.
"""

import argparse
import csv
from pathlib import Path

MODEL_NAMES = {
    "gemma3-12b": "Gemma 3 12B (RANA)", "gemma3-4b": "Gemma 3 4B", "gemma3-1b": "Gemma 3 1B",
    "gemma3-270m": "Gemma 3 270M", "gemma2-9b": "Gemma 2 9B", "qwen3-8b": "Qwen3 8B",
    "qwen3.5-9b": "Qwen3.5 9B", "qwen3.5-2b": "Qwen3.5 2B", "qwen2.5-14b": "Qwen2.5 14B",
    "llama3.2-1b": "Llama 3.2 1B", "deepseek-r1-llama-8b": "DeepSeek-R1-Distill-Llama-8B",
}
FORMAT_NAMES = {"label_only": "Label only", "label_first": "Label first",
                "reasoning_first": "Reasoning first (RANA)", "assessment_only": "Assessment only"}
ABLATION_NAMES = {
    "rag_k5_random": "Random neighbors", "rag_k5_flipped": "Neighbor labels inverted",
    "rag_k5_nolabels": "Neighbor labels hidden", "rag_k5_notext": "Neighbor text hidden",
    "rag_k5_maxsim0.90": "Neighbors above cosine 0.90 removed",
    "rag_k5_lang-cross": "Other-language neighbors only", "rag_k5_minilm": "MiniLM index instead of bge-m3",
}


def load(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pc(v):
    return f"{100 * float(v):.2f}" if v not in (None, "") else "--"


def pval(v):
    if v in (None, ""):
        return "--"
    v = float(v)
    if v < 1e-15:  # below double precision: report a bound, not 0
        return "$<10^{-15}$"
    if v < 1e-3:
        m, e = f"{v:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{v:.3f}"


def esc(s):
    return s.replace("_", "\\_").replace("%", "\\%")


def table(caption, label, header, rows, colspec, wide=True, note=None):
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[t]"]
    if wide:
        lines.append("\\renewcommand{\\tblwidth}{\\textwidth}%")
    lines += ["\\small", "\\centering", f"\\caption{{{caption}}}\\label{{{label}}}",
              f"\\begin{{tabular*}}{{\\tblwidth}}{{@{{\\extracolsep{{\\fill}}}}{colspec}@{{}}}}", "\\toprule",
              " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular*}"]
    if note:
        lines += ["", "\\vspace{0.4em}", f"\\footnotesize {note}"]
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines) + "\n"


def by_run(metrics):
    return {m["run"]: m for m in metrics}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-dir", default="revision/report")
    ap.add_argument("--out-dir", default="revision/report/latex")
    a = ap.parse_args()
    R, out = Path(a.report_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def save(name, text):
        (out / f"{name}.tex").write_text(text, encoding="utf-8")
        written.append(name)

    # 1. Main table: every model, Non-RAG vs RAG, test and withheld publisher
    met = by_run(load(R / "main/metrics.csv"))
    mcn = {(r["A"], r["B"]): r for r in load(R / "main/mcnemar.csv")}
    rows = []
    for key, name in MODEL_NAMES.items():
        base = f"{key}__reasoning_first"
        t0, t1 = met.get(f"{base}/test__norag"), met.get(f"{base}/test__rag_k5")
        x0, x1 = met.get(f"{base}/test_xsource__norag"), met.get(f"{base}/test_xsource__rag_k5")
        if not (t0 or t1):
            continue
        mt = mcn.get((f"{base}/test__norag", f"{base}/test__rag_k5"), {})
        mx = mcn.get((f"{base}/test_xsource__norag", f"{base}/test_xsource__rag_k5"), {})
        rows.append([name, pc(t0 and t0["accuracy"]), pc(t1 and t1["accuracy"]), pc(t1 and t1["macro_f1"]),
                     pval(mt.get("p_holm")), pc(x0 and x0["accuracy"]), pc(x1 and x1["accuracy"]), pval(mx.get("p_holm"))])
    if rows:
        save("tbl_main", table(
            "Accuracy (\\%) without and with retrieval on the locked test partition and on the withheld publisher, with macro-F1 for the retrieval condition and Holm-adjusted McNemar $p$ for the retrieval effect.",
            "tbl4", ["Model", "Test", "Test RAG", "Macro-F1", "$p$", "Withheld", "Withheld RAG", "$p$"], rows, "LRRRRRRR"))

    # 2. Language breakdown
    bl = load(R / "main/by_lang.csv")
    rows = []
    for key, name in MODEL_NAMES.items():
        run = f"{key}__reasoning_first/test__rag_k5"
        d = {r["lang"]: r for r in bl if r["run"] == run}
        if "id" in d and "en" in d:
            g = d.get("id_minus_en", {})
            rows.append([name, f"{pc(d['id']['accuracy'])} [{pc(d['id']['acc_ci_low'])}, {pc(d['id']['acc_ci_high'])}]",
                         f"{pc(d['en']['accuracy'])} [{pc(d['en']['acc_ci_low'])}, {pc(d['en']['acc_ci_high'])}]",
                         pval(g.get("fisher_p"))])
    if rows:
        save("tbl_lang", table("Accuracy (\\%, Wilson 95\\% interval) by language on the test partition with retrieval, and Fisher's exact test for the difference.",
                               "tbl_lang", ["Model", "Indonesian", "English", "Fisher $p$"], rows, "LRRR"))

    # 3. Baselines against RANA
    bm = by_run(load(R / "baselines/metrics.csv"))
    spec = [("knn/minilm/test__weighted_k5", "knn/minilm/test_xsource__weighted_k5", "kNN, MiniLM, weighted $k=5$"),
            ("knn/e5-large/test__weighted_k5", "knn/e5-large/test_xsource__weighted_k5", "kNN, e5-large, weighted $k=5$"),
            ("knn/bge-m3/test__weighted_k5", "knn/bge-m3/test_xsource__weighted_k5", "kNN, bge-m3, weighted $k=5$"),
            ("encoders__indobert-base-p1/test", "encoders__indobert-base-p1/test_xsource", "IndoBERT"),
            ("encoders__bert-base-multilingual-cased/test", "encoders__bert-base-multilingual-cased/test_xsource", "Multilingual BERT"),
            ("encoders__xlm-roberta-base/test", "encoders__xlm-roberta-base/test_xsource", "XLM-RoBERTa"),
            ("zeroshot__gemma3-12b/test__norag", None, "Gemma 3 12B, zero-shot"),
            ("gemma3-12b__reasoning_first/test__norag", "gemma3-12b__reasoning_first/test_xsource__norag", "RANA without retrieval"),
            ("gemma3-12b__reasoning_first/test__rag_k5", "gemma3-12b__reasoning_first/test_xsource__rag_k5", "RANA")]
    rows = []
    for t, x, name in spec:
        mt, mx = bm.get(t), bm.get(x) if x else None
        if mt or mx:
            rows.append([name, pc(mt and mt["accuracy"]), pc(mt and mt["macro_f1"]), pc(mx and mx["accuracy"]), pc(mx and mx["macro_f1"])])
    if rows:
        save("tbl_baselines_models", table("RANA against trained baselines on the same partition: accuracy and macro-F1 (\\%) on the test partition and the withheld publisher.",
                                           "tbl_sota", ["System", "Test acc.", "Test F1", "Withheld acc.", "Withheld F1"], rows, "LRRRR"))

    # 4. Retrieval controls and k sweep (paired on the same 1,000 test articles)
    am = load(R / "ablation/mcnemar.csv")
    rows, krows = [], {}
    for r in am:
        a_run, b_run = r["A"], r["B"]
        model = a_run.split("__")[0]
        cond = b_run.split("/test__")[-1]
        if cond in ABLATION_NAMES and model == "gemma3-12b":
            rows.append([ABLATION_NAMES[cond], pc(r["acc_A"]), pc(r["acc_B"]),
                         f"{100 * float(r['diff_B_minus_A']):+.2f} [{100 * float(r['diff_ci_low']):+.2f}, {100 * float(r['diff_ci_high']):+.2f}]",
                         pval(r["p_holm"])])
        if cond.startswith("rag_k") and cond[5:].isdigit():
            krows.setdefault(model, {"5": r["acc_A"]})[cond[5:]] = r["acc_B"]
    if rows:
        save("tbl_ablation", table("Retrieval controls for RANA on the same 1,000 test articles: accuracy (\\%) with standard retrieval, under the control, the paired difference with its 95\\% bootstrap interval, and Holm-adjusted McNemar $p$.",
                                   "tbl_ablation", ["Condition", "Standard", "Control", "Difference", "$p$"], rows, "LRRRR"))
    if krows:
        ks = sorted({k for v in krows.values() for k in v}, key=int)
        rows = [[MODEL_NAMES.get(m, m)] + [pc(v.get(k)) for k in ks] for m, v in krows.items()]
        save("tbl_ksweep", table("Accuracy (\\%) by retrieval depth $k$ on the same 1,000 test articles.",
                                 "tbl_ksweep", ["Model"] + [f"$k={k}$" for k in ks], rows, "L" + "R" * len(ks), wide=False))

    # 5. Output-format factorial
    fm = by_run(load(R / "factorial/metrics.csv"))
    fs = load(R / "factorial/signals.csv")
    fmodel = next((r["run"].split("__")[0] for r in fs if "__label_only/" in r["run"]), "gemma3-4b")
    rows = []
    for f, fname in FORMAT_NAMES.items():
        base = f"{fmodel}__{f}"
        n0, n1 = fm.get(f"{base}/test__norag"), fm.get(f"{base}/test__rag_k5")
        sig = [r for r in fs if r["run"] == f"{base}/test__rag_k5" and r["signal"] not in ("label_matches_rule_on_own_signals",)]
        cons = next((r for r in fs if r["run"] == f"{base}/test__rag_k5" and r["signal"] == "label_matches_rule_on_own_signals"), None)
        sig_f1 = sum(float(r["macro_f1"]) for r in sig) / len(sig) if sig else None
        if n0 or n1:
            rows.append([fname, pc(n0 and n0["accuracy"]), pc(n1 and n1["accuracy"]), pc(n1 and n1["macro_f1"]),
                         pc(sig_f1), pc(cons and cons["accuracy"])])
    if rows:
        save("tbl_factorial", table(f"Output formats trained on {MODEL_NAMES.get(fmodel, fmodel)} with identical data and settings: accuracy (\\%) without and with retrieval, macro-F1, mean signal macro-F1 against annotators, and how often the label follows the all-four rule applied to the model's own signals.",
                                    "tbl_factorial", ["Format", "Acc.", "Acc. RAG", "Macro-F1", "Signal F1", "Label = rule"], rows, "LRRRRR"))

    # 6. Signal agreement for RANA
    sg = load(R / "main/signals.csv")
    rows = []
    for s in ("positive_tone", "persuasive", "brand_promotion", "single_perspective"):
        t = next((r for r in sg if r["run"] == "gemma3-12b__reasoning_first/test__rag_k5" and r["signal"] == s), None)
        x = next((r for r in sg if r["run"] == "gemma3-12b__reasoning_first/test_xsource__rag_k5" and r["signal"] == s), None)
        if t or x:
            label = {"positive_tone": "Positive tone", "persuasive": "Persuasive language",
                     "brand_promotion": "Brand promotion", "single_perspective": "Single perspective"}[s]
            rows.append([label, pc(t and t["accuracy"]), pc(t and t["macro_f1"]), pc(x and x["accuracy"]), pc(x and x["macro_f1"])])
    if rows:
        save("tbl_signals", table("Agreement (\\%) of the signals RANA reports with the annotators: accuracy and macro-F1 on the test partition and the withheld publisher.",
                                  "tbl_signals", ["Signal", "Test acc.", "Test F1", "Withheld acc.", "Withheld F1"], rows, "LRRRR", wide=False))

    # 7. Faithfulness
    fa = load(R / "faithfulness/faithfulness.csv")
    names = {"all_yes": "All four set present", "all_no": "All four set absent",
             "flip_positive_tone": "Positive tone flipped", "flip_persuasive": "Persuasive language flipped",
             "flip_brand_promotion": "Brand promotion flipped", "flip_single_perspective": "Single perspective flipped"}
    rows = [[names.get(r["variant"], esc(r["variant"])), r["n"], pc(r["label_follows_rule"]), pc(r["label_changed"])] for r in fa]
    if rows:
        save("tbl_faithfulness", table("Signal interventions on RANA: share (\\%) of outputs whose label then agrees with the all-four rule applied to the forced signals, and share whose label changed.",
                                       "tbl_faithfulness", ["Intervention", "$n$", "Label follows rule", "Label changed"], rows, "LRRR", wide=False))

    print(f"written to {out}: {', '.join(written) if written else 'nothing (no report CSVs found)'}")


if __name__ == "__main__":
    main()
