"""
LLM-as-a-Judge and its validation against human editors (R1-C6, R3, minor 7).

The API key is read from the OPENROUTER_API_KEY environment variable; never put it on the command line.

1) Judge a NATURAL random sample (estimates average quality) plus a separate ERROR sample:
    python revision/judge.py run --pred revision/results/gemma3-12b/rag_k5.jsonl \
        --test-file revision/splits/main/test.jsonl --out revision/judge/gemma3-12b_rag_k5.jsonl \
        --n-natural 150 --n-errors 50

2) Export a blinded rating sheet for human assessors (same items the judge saw, models mixed and shuffled):
    python revision/judge.py export-human --judged revision/judge/*.jsonl --n 120 \
        --out revision/judge/human_sheet.csv

   Give each assessor a copy (rater_A.csv, rater_B.csv, rater_C.csv). They fill every score column with 1-5
   (5 = best on every criterion, so no_hallucination 5 = no invented content) without seeing model names.

3) Agreement among humans and between humans and the judge:
    python revision/judge.py agreement --human rater_A.csv rater_B.csv rater_C.csv \
        --key revision/judge/human_sheet.key.json --judged "revision/judge/*.jsonl" --out revision/judge/agreement.json
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SIGNALS, append_jsonl, read_jsonl, write_json

CRITERIA = {
    "grounding": "Every claim in the assessment is supported by the article text (factual grounding).",
    "signal_correctness": "Each of the four signals is identified correctly for this article.",
    "completeness": "All four signals are addressed with a usable judgment.",
    "specificity": "The assessment refers to concrete features of this article, not generic statements.",
    "editorial_usefulness": "An editor could act on this assessment when deciding whether to flag the article.",
    "no_hallucination": "The assessment contains no invented facts, quotes, brands or sources (5 = none at all).",
    "label_consistency": "The final label follows from the stated signals under the codebook rule.",
}

JUDGE_PROMPT = """You are an experienced news editor auditing an automated assistant that flags native advertising
(promotional content written to look like news). Codebook: an article is native ads only when all four signals are
present together: positive_tone, persuasive, brand_promotion, single_perspective.

ARTICLE:
{article}

ASSISTANT OUTPUT:
{output}

Score each criterion from 1 (very poor) to 5 (excellent). Judge the assessment against the article itself;
you are not told the gold label.
{criteria}

Answer only with JSON: {{{keys}, "comment": "<one sentence>"}}"""


def call_openrouter(model, prompt, retries=5):
    import requests
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("Set OPENROUTER_API_KEY in the environment.")
    for attempt in range(retries):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=120,
                              headers={"Authorization": f"Bearer {key}"},
                              json={"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}],
                                    "response_format": {"type": "json_object"}})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            wait = 2 ** attempt
            print(f"   API error ({e}); retry in {wait}s")
            time.sleep(wait)
    return None


def output_text(p):
    obj = {}
    if p.get("signals_pred"):
        obj["signals"] = p["signals_pred"]
    if p.get("evidence_pred"):
        obj["evidence"] = p["evidence_pred"]
    obj["label"] = p.get("label_pred")
    return json.dumps(obj, ensure_ascii=False)


def cmd_run(a):
    preds = read_jsonl(a.pred)
    text = {r["id"]: r["text"] for r in read_jsonl(a.test_file)}
    rng = random.Random(a.seed)
    natural = rng.sample(preds, min(a.n_natural, len(preds)))
    chosen = {p["id"] for p in natural}
    errors = [p for p in preds if p["label_gold"] != p.get("label_pred") and p["id"] not in chosen]
    errors = rng.sample(errors, min(a.n_errors, len(errors)))
    items = [("natural", p) for p in natural] + [("error", p) for p in errors]
    done = {r["id"] for r in read_jsonl(a.out)} if Path(a.out).exists() else set()
    crit = "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())
    keys = ", ".join(f'"{k}": <1-5>' for k in CRITERIA)
    for i, (sample, p) in enumerate(items):
        if p["id"] in done:
            continue
        raw = call_openrouter(a.model, JUDGE_PROMPT.format(article=text[p["id"]][:3000], output=output_text(p),
                                                           criteria=crit, keys=keys))
        try:
            scores = json.loads(raw)
        except Exception:
            scores = None
        append_jsonl(a.out, {"id": p["id"], "sample": sample, "run": a.pred, "judge_model": a.model,
                             "label_gold": p["label_gold"], "label_pred": p.get("label_pred"), "lang": p["lang"],
                             "output": output_text(p), "scores": scores, "raw": raw})
        print(f"[{i + 1}/{len(items)}] {sample}")
    rows = read_jsonl(a.out)
    for sample in ("natural", "error"):
        sub = [r for r in rows if r["sample"] == sample and r["scores"]]
        if sub:
            print(sample, {k: round(float(np.mean([r["scores"].get(k, np.nan) for r in sub])), 3) for k in CRITERIA})


def cmd_export(a):
    judged = [r for pat in a.judged for f in glob.glob(pat) for r in read_jsonl(f) if r["scores"]]
    rng = random.Random(a.seed)
    natural = [r for r in judged if r["sample"] == "natural"]
    items = rng.sample(natural, min(a.n, len(natural)))
    test = {r["id"]: r["text"] for r in read_jsonl(a.test_file)}
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "article", "assistant_output"] + list(CRITERIA) + ["comment"])
        key = []
        for n, r in enumerate(items, 1):
            w.writerow([n, test[r["id"]][:3000], r["output"]] + [""] * len(CRITERIA) + [""])
            key.append({"item": n, "id": r["id"], "run": r["run"]})
    write_json(Path(a.out).with_suffix(".key.json"), key)
    print(f"✅ {len(items)} items -> {a.out} (unblinding key: {Path(a.out).with_suffix('.key.json')}, keep it from raters)")


def kalpha_interval(matrix):
    """Krippendorff's alpha, interval metric. matrix: raters x items, np.nan = missing."""
    m = np.asarray(matrix, dtype=float)
    units = [m[:, j][~np.isnan(m[:, j])] for j in range(m.shape[1])]
    units = [u for u in units if len(u) >= 2]
    n = sum(len(u) for u in units)
    if n < 2:
        return float("nan")
    do = sum(((u[:, None] - u[None, :]) ** 2).sum() / (len(u) - 1) for u in units) / n
    allv = np.concatenate(units)
    de = ((allv[:, None] - allv[None, :]) ** 2).sum() / (n * (n - 1))
    return 1 - do / de if de else float("nan")


def cmd_agreement(a):
    from scipy.stats import spearmanr
    key = json.load(open(Path(a.key)))
    raters = []
    for f in a.human:
        with open(f, encoding="utf-8") as fh:
            raters.append({int(r["item"]): r for r in csv.DictReader(fh)})
    judged = {(r["id"], r["run"]): r for pat in a.judged for f in glob.glob(pat) for r in read_jsonl(f)}
    result = {}
    for c in CRITERIA:
        mat = [[float(rt[k["item"]][c]) if rt.get(k["item"]) and rt[k["item"]][c].strip() else np.nan
                for k in key] for rt in raters]
        human_mean = np.nanmean(mat, axis=0)
        judge = np.array([float(judged[(k["id"], k["run"])]["scores"].get(c, np.nan)) for k in key])
        ok = ~np.isnan(human_mean) & ~np.isnan(judge)
        result[c] = {"alpha_humans_interval": kalpha_interval(mat),
                     "alpha_humans_plus_judge": kalpha_interval(mat + [judge.tolist()]),
                     "spearman_judge_vs_human_mean": float(spearmanr(judge[ok], human_mean[ok]).statistic) if ok.sum() > 2 else None,
                     "mean_human": float(np.nanmean(human_mean)), "mean_judge": float(np.nanmean(judge)),
                     "n_items": int(ok.sum()), "n_raters": len(raters)}
    write_json(a.out, result)
    for c, v in result.items():
        print(f"{c:22s} α_humans={v['alpha_humans_interval']:.3f}  α_with_judge={v['alpha_humans_plus_judge']:.3f}  "
              f"ρ(judge,human)={v['spearman_judge_vs_human_mean']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pred", required=True)
    r.add_argument("--test-file", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n-natural", type=int, default=150)
    r.add_argument("--n-errors", type=int, default=50)
    r.add_argument("--model", default="openai/gpt-4o-mini")
    r.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("export-human")
    e.add_argument("--judged", nargs="+", required=True)
    e.add_argument("--test-file", default="revision/splits/main/test.jsonl")
    e.add_argument("--n", type=int, default=120)
    e.add_argument("--out", required=True)
    e.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("agreement")
    g.add_argument("--human", nargs="+", required=True)
    g.add_argument("--key", required=True, help="the .key.json written by export-human")
    g.add_argument("--judged", nargs="+", required=True)
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "export-human": cmd_export, "agreement": cmd_agreement}[a.cmd](a)


if __name__ == "__main__":
    main()
