"""
OPTIONAL: per-signal evidence sentences for train/val, used as the "evidence" field of the target.

The first-submission training data had only four distinct "reasoning" strings (one per
label x language), so the fine-tuned models never learned article-specific explanations.
This script writes one short, article-grounded justification per signal. The annotators'
signal values are GIVEN to the generator, so the evidence explains the human annotation
rather than inventing a new one. Disclose it in the paper as LLM-written, annotation-
conditioned rationales (distillation), and spot-check a sample by hand.

    export OPENROUTER_API_KEY=...        # never pass the key on the command line
    python revision/generate_rationales.py --split-dir revision/splits/main \
        --out revision/splits/main/evidence.jsonl --model openai/gpt-4o-mini

Only train.jsonl and val.jsonl are processed; test articles are never sent.
Resumable: re-run the same command after an interruption.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SIGNALS, append_jsonl, read_jsonl
from judge import call_openrouter

PROMPT = {
    "id": """Berikut artikel berita beserta penilaian empat sinyal oleh anotator manusia.
Tulis satu kalimat bukti (maks. 25 kata) untuk setiap sinyal yang menjelaskan MENGAPA nilai tersebut tepat,
dengan merujuk pada isi artikel (kutip frasa pendek bila ada). Jangan menambahkan fakta di luar artikel.

Penilaian anotator: {signals}

Artikel:
{text}

Jawab hanya dengan JSON: {{"positive_tone": "...", "persuasive": "...", "brand_promotion": "...", "single_perspective": "..."}}""",
    "en": """Below is a news article and the four-signal assessment made by human annotators.
Write one evidence sentence (max. 25 words) per signal explaining WHY that value is right, referring to the
article's content (quote a short phrase when possible). Do not add facts that are not in the article.

Annotator assessment: {signals}

Article:
{text}

Answer only with JSON: {{"positive_tone": "...", "persuasive": "...", "brand_promotion": "...", "single_perspective": "..."}}""",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--max-chars", type=int, default=1500, help="Same article window the fine-tuned model sees")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.split_dir) / "train.jsonl") + read_jsonl(Path(args.split_dir) / "val.jsonl")
    done = {r["id"] for r in read_jsonl(args.out)} if Path(args.out).exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} train+val articles, {len(todo)} to go")
    for i, r in enumerate(todo, 1):
        lang = "en" if r["lang"] == "en" else "id"
        raw = call_openrouter(args.model, PROMPT[lang].format(signals=json.dumps(r["signals"], ensure_ascii=False),
                                                              text=r["text"][: args.max_chars]))
        try:
            ev = json.loads(raw)
            ev = {s: str(ev[s]).strip() for s in SIGNALS}
        except Exception:
            print(f"   invalid output for id {r['id']}, will retry on next run")
            continue
        append_jsonl(args.out, {"id": r["id"], "evidence": ev, "generator": args.model})
        if i % 100 == 0:
            print(f"[{i}/{len(todo)}]")
    print(f"✅ -> {args.out}")


if __name__ == "__main__":
    main()
