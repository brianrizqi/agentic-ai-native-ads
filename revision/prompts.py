"""
One prompt, one target format, shared by training and inference for every model (R1-C11).

Output formats (R1-C5 / R2 factorial; only the JSON key order and the keys present change,
the instruction text and the information in the prompt are identical across conditions):

    label_only       {"label"}
    label_first      {"label", "signals", ["evidence"]}
    reasoning_first  {"signals", ["evidence"], "label"}        <- RANA
    assessment_only  {"signals", ["evidence"]}                 label = codebook rule over signals

"signals" are the four annotator-coded indicators from native_ads_dataset.xlsx.
"evidence" (optional) is a short per-signal justification from generate_rationales.py.
"""

import json
import re
from typing import Dict, List, Optional

from common import NATIVE, NEWS, SIGNALS, label_from_signals, norm_label, norm_signal

FORMATS = ("label_only", "label_first", "reasoning_first", "assessment_only")

LABEL_TEXT = {"id": {NATIVE: "native ads", NEWS: "berita murni"},
              "en": {NATIVE: "native ads", NEWS: "pure news"}}
VALUE_TEXT = {"id": {"ya": "ya", "netral": "netral", "tidak": "tidak"},
              "en": {"ya": "yes", "netral": "neutral", "tidak": "no"}}

INSTRUCTION = {
    "id": """Anda adalah asisten editor. Tentukan apakah artikel berikut merupakan "native ads" (konten promosi yang ditulis menyerupai berita) atau "berita murni".

Nilai empat sinyal berikut berdasarkan isi artikel:
1. positive_tone: nada terhadap subjek yang diberitakan, "ya" (memuji/positif), "netral", atau "tidak" (kritis/negatif).
2. persuasive: bahasa yang mengajak atau meyakinkan pembaca, "ya" atau "tidak".
3. brand_promotion: artikel menonjolkan produk, merek, atau lembaga tertentu secara menguntungkan, "ya" atau "tidak".
4. single_perspective: hanya satu sumber atau sudut pandang yang disajikan, "ya" atau "tidak".

Artikel dilabeli "native ads" hanya jika keempat sinyal hadir bersama. Menyebut nama merek saja tidak cukup: ulasan produk, opini, atau berita korporasi yang memuat kritik atau lebih dari satu sudut pandang tetap "berita murni".""",
    "en": """You are an editorial assistant. Decide whether the following article is "native ads" (promotional content written to look like news) or "pure news".

Assess four signals from the article content:
1. positive_tone: tone toward the subject, "yes" (praising/positive), "neutral", or "no" (critical/negative).
2. persuasive: language that urges or convinces the reader, "yes" or "no".
3. brand_promotion: the article presents a specific product, brand, or institution favorably, "yes" or "no".
4. single_perspective: only one source or viewpoint is presented, "yes" or "no".

Label an article "native ads" only when all four signals are present together. Mentioning a brand is not enough: product reviews, opinion pieces, or corporate news that include criticism or more than one viewpoint remain "pure news".""",
}

REF_HEADER = {"id": "Artikel referensi dari data latih berlabel (gunakan sebagai pembanding, bukan untuk disalin):",
              "en": "Reference articles from the labeled training data (use for comparison, not to copy):"}
REF_HIDDEN_TEXT = {"id": "[teks disembunyikan]", "en": "[text hidden]"}
REF_HIDDEN_LABEL = {"id": "tidak diketahui", "en": "unknown"}
ARTICLE_HEADER = {"id": "Artikel yang dinilai:", "en": "Article to assess:"}


def _schema_example(fmt: str, lang: str, use_evidence: bool) -> str:
    v, l = VALUE_TEXT[lang], LABEL_TEXT[lang]
    sig = "{" + ", ".join(f'"{s}": "{v["ya"]}|{v["tidak"]}' + (f'|{v["netral"]}' if s == "positive_tone" else "") + '"'
                          for s in SIGNALS) + "}"
    ev = "{" + ", ".join(f'"{s}": "..."' for s in SIGNALS) + "}"
    lab = f'"{l[NATIVE]}|{l[NEWS]}"'
    parts = {
        "label_only": [f'"label": {lab}'],
        "label_first": [f'"label": {lab}', f'"signals": {sig}'] + ([f'"evidence": {ev}'] if use_evidence else []),
        "reasoning_first": [f'"signals": {sig}'] + ([f'"evidence": {ev}'] if use_evidence else []) + [f'"label": {lab}'],
        "assessment_only": [f'"signals": {sig}'] + ([f'"evidence": {ev}'] if use_evidence else []),
    }[fmt]
    lead = "Jawab hanya dengan JSON berikut:" if lang == "id" else "Answer only with this JSON:"
    return lead + "\n{" + ", ".join(parts) + "}"


def render_neighbors(neighbors: List[dict], lang: str, show_label=True, show_text=True,
                     show_signals=False, ref_chars=400) -> str:
    if not neighbors:
        return ""
    lines = [REF_HEADER[lang]]
    for i, nb in enumerate(neighbors, 1):
        lab = LABEL_TEXT[lang][nb["label"]] if show_label else REF_HIDDEN_LABEL[lang]
        head = f"[{i}] label: {lab}"
        if show_signals and show_label:
            head += "; " + ", ".join(f"{s}={VALUE_TEXT[lang][nb['signals'][s]]}" for s in SIGNALS)
        body = nb["text"][:ref_chars].replace("\n", " ") if show_text else REF_HIDDEN_TEXT[lang]
        lines.append(f"{head}\n{body}")
    return "\n\n".join(lines)


def build_prompt(text: str, lang: str, fmt: str, use_evidence: bool = False, neighbors: Optional[List[dict]] = None,
                 max_chars: int = 1500, **render_kw) -> str:
    lang = "en" if lang == "en" else "id"
    blocks = [INSTRUCTION[lang]]
    ref = render_neighbors(neighbors or [], lang, **render_kw)
    if ref:
        blocks.append(ref)
    blocks.append(f"{ARTICLE_HEADER[lang]}\n{text[:max_chars]}")
    blocks.append(_schema_example(fmt, lang, use_evidence))
    return "\n\n".join(blocks)


def _signals_text(signals: Dict[str, str], lang: str) -> Dict[str, str]:
    return {s: VALUE_TEXT[lang][signals[s]] for s in SIGNALS}


def build_target(rec: dict, fmt: str, use_evidence: bool = False) -> str:
    lang = "en" if rec["lang"] == "en" else "id"
    sig = _signals_text(rec["signals"], lang)
    lab = LABEL_TEXT[lang][rec["label"]]
    ev = rec.get("evidence") if use_evidence else None
    if use_evidence and not ev:
        raise ValueError(f"record {rec['id']} has no evidence but use_evidence=True")
    obj = {
        "label_only": lambda: {"label": lab},
        "label_first": lambda: {"label": lab, "signals": sig, **({"evidence": ev} if ev else {})},
        "reasoning_first": lambda: {"signals": sig, **({"evidence": ev} if ev else {}), "label": lab},
        "assessment_only": lambda: {"signals": sig, **({"evidence": ev} if ev else {})},
    }[fmt]()
    return json.dumps(obj, ensure_ascii=False)


def forced_signal_prefix(signals: Dict[str, str], lang: str) -> str:
    """Assistant-turn prefix that fixes the signals, for faithfulness interventions (R1-C5).

    Byte-identical to the start of build_target(..., 'reasoning_first'), so the model
    continues exactly as it would after generating these signals itself.
    """
    lang = "en" if lang == "en" else "id"
    return '{"signals": ' + json.dumps(_signals_text(signals, lang), ensure_ascii=False) + ", "


def apply_chat(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    try:  # Qwen3: disable the thinking block so every model answers in the same format
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------- parsing

def _first_json_object(s: str) -> Optional[dict]:
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", s):
        try:
            obj, _ = dec.raw_decode(s[m.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def parse_output(raw: str, fmt: str) -> dict:
    """Parse a generation. Invalid outputs are kept and counted, never silently relabeled (R1 Q15).

    label_source: 'json' (valid JSON), 'regex' (label recovered from malformed JSON),
                  'rule' (assessment_only: codebook rule over parsed signals), 'invalid'.
    """
    s = re.sub(r"<think>.*?</think>", "", str(raw), flags=re.DOTALL).strip()
    obj = _first_json_object(s)
    out = {"label": None, "signals": None, "evidence": None, "parse_ok": obj is not None, "label_source": "invalid"}
    if obj:
        sig = obj.get("signals")
        if isinstance(sig, dict):
            parsed = {k: norm_signal(k, sig.get(k)) for k in SIGNALS}
            out["signals"] = parsed if all(parsed.values()) else None
        if isinstance(obj.get("evidence"), dict):
            out["evidence"] = obj["evidence"]
        if fmt == "assessment_only":
            out["label"] = label_from_signals(out["signals"]) if out["signals"] else None
            out["label_source"] = "rule" if out["label"] else "invalid"
        else:
            out["label"] = norm_label(obj.get("label"))
            out["label_source"] = "json" if out["label"] else "invalid"
    if out["label"] is None and fmt != "assessment_only":
        m = re.search(r'"label"\s*:\s*"([^"]+)"', s)
        if m and norm_label(m.group(1)):
            out["label"], out["label_source"] = norm_label(m.group(1)), "regex"
    return out
