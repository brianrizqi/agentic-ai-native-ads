"""Checks the judge against facts that are known exactly for every judged output.

signal_correctness is compared with the number of the four signals that match the annotators,
and label_consistency with whether the label follows the all-four rule applied to the model's
own signals. A judge criterion that does not track these is not used as evidence in the paper.
Prints aggregates only (no article text).

    python revision/judge_validity.py --judged "revision/judge/*__rag_k5.jsonl" --out revision/judge/validity.json
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SIGNALS, label_from_signals, read_jsonl, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = {}
    for f in sorted(glob.glob(a.judged)):
        rows = [r for r in read_jsonl(f) if isinstance(r.get("scores"), dict)]
        preds = {p["id"]: p for p in read_jsonl(rows[0]["run"])}
        out = {}
        for sample in ("natural", "error"):
            sig, n_match, lc, rule_ok = [], [], [], []
            for r in rows:
                p = preds.get(r["id"])
                if r["sample"] != sample or not p or not p.get("signals_pred"):
                    continue
                sp, sg = p["signals_pred"], p["signals_gold"]
                sig.append(r["scores"].get("signal_correctness"))
                n_match.append(sum(sp.get(s) == sg.get(s) for s in SIGNALS))
                lc.append(r["scores"].get("label_consistency"))
                rule_ok.append(int(label_from_signals(sp) == p["label_pred"]))
            if not sig:
                continue
            rho_sig = spearmanr(sig, n_match)
            rho_lc = spearmanr(lc, rule_ok) if len(set(rule_ok)) > 1 else None
            out[sample] = {
                "n": len(sig),
                "judge_signal_correctness_mean": round(float(np.mean(sig)), 3),
                "signals_matching_annotators_mean_of_4": round(float(np.mean(n_match)), 3),
                "spearman_signal_correctness_vs_matches": round(float(rho_sig.statistic), 3),
                "p_signal": float(rho_sig.pvalue),
                "judge_label_consistency_mean": round(float(np.mean(lc)), 3),
                "label_follows_rule_share": round(float(np.mean(rule_ok)), 3),
                "spearman_label_consistency_vs_rule": None if rho_lc is None else round(float(rho_lc.statistic), 3),
                "label_consistency_mean_when_rule_holds": round(float(np.mean([x for x, o in zip(lc, rule_ok) if o])), 3)
                if any(rule_ok) else None,
            }
        report[Path(f).name] = out
        print(f"\n== {Path(f).name}")
        for sample, v in out.items():
            print(f"  {sample}: " + ", ".join(f"{k}={v[k]}" for k in v))
    write_json(a.out, report)


if __name__ == "__main__":
    main()
