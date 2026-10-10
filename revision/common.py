"""
Shared helpers for the leakage-free revision pipeline (IEEE Access resubmission).

Every script in revision/ reads and writes JSONL records with the same schema:

    id, text, lang, label, signals{4}, url, host, outlet, section, year,
    cluster_id, split

so predictions from any model (LLM, kNN, encoder baselines) can be joined by id.
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

NATIVE, NEWS = "native ads", "berita murni"
LABELS = (NATIVE, NEWS)

# The four journalistic signals, keyed by the column names in native_ads_dataset.xlsx.
SIGNALS = ("positive_tone", "persuasive", "brand_promotion", "single_perspective")
SIGNAL_COLUMNS = {
    "positive_tone": "label-positif",
    "persuasive": "label-persuasif",
    "brand_promotion": "label-menggambarkan-produk",
    "single_perspective": "label-sumber-berita-tunggal",
}
SIGNAL_VALUES = {
    "positive_tone": ("ya", "netral", "tidak"),
    "persuasive": ("ya", "tidak"),
    "brand_promotion": ("ya", "tidak"),
    "single_perspective": ("ya", "tidak"),
}


# ---------------------------------------------------------------- labels

def norm_label(value) -> Optional[str]:
    s = str(value or "").strip().lower()
    # A copied schema line ("native ads|berita murni") names both classes: not an answer.
    if "|" in s or ("native" in s and ("murni" in s or "news" in s)):
        return None
    if "native" in s or s in ("ads", "iklan", "a"):
        return NATIVE
    if "murni" in s or "news" in s or "berita" in s or s == "b":
        return NEWS
    return None


def norm_signal(name: str, value) -> Optional[str]:
    s = str(value or "").strip().lower()
    aliases = {"yes": "ya", "y": "ya", "true": "ya", "no": "tidak", "n": "tidak",
               "false": "tidak", "neutral": "netral"}
    s = aliases.get(s, s)
    return s if s in SIGNAL_VALUES[name] else None


def label_from_signals(signals: Dict[str, str]) -> Optional[str]:
    """Deterministic codebook rule: Native Ads only when all four signals are present.

    Used by the assessment_only condition. Returns None if any signal is missing.
    """
    if not signals or any(signals.get(s) is None for s in SIGNALS):
        return None
    return NATIVE if all(signals[s] == "ya" for s in SIGNALS) else NEWS


# ---------------------------------------------------------------- source metadata

def parse_url(url: str) -> Dict[str, Optional[str]]:
    host = urlparse(str(url)).netloc.lower().replace("www.", "")
    parts = host.split(".")
    outlet = ".".join(parts[-3:]) if host.endswith(".co.id") else ".".join(parts[-2:])
    path = urlparse(str(url)).path.strip("/").split("/")
    # thestar.com/<section>/..., kompas.com/<channel>/read/..., biz.kompas.com, food.detik.com
    if outlet == "thestar.com":
        section = path[0] if path else ""
    else:
        sub = host[: -len(outlet)].rstrip(".")
        section = sub or (path[0] if path else "")
    year = re.search(r"/(20\d\d)/\d\d/", str(url))
    return {"host": host, "outlet": outlet, "section": section,
            "year": year.group(1) if year else None}


# ---------------------------------------------------------------- text

def normalize_for_hash(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Source cues that identify the outlet or wire service rather than the content.
# Stripped from every article (train, index and test) so that a model cannot
# classify by dateline format, e.g. "(GLOBE NEWSWIRE) —" vs "(AP) —".
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_LEADING_CUES = [
    # Press-release legal header: "NOT FOR DISSEMINATION IN THE UNITED STATES ..."
    r"(?i)^[\s—–-]*NOT FOR (?:DISSEMINATION|DISTRIBUTION)[^\n]*\n+",
    # EN wire/agency datelines: "TORONTO, March 13, 2026 (GLOBE NEWSWIRE) — "
    r"^[^\n]{0,120}?\((?:GLOBE NEWSWIRE|GLOBENEWSWIRE|AP|Reuters|CP|AFP|Bloomberg|PRNewswire|"
    r"Business Wire|ACCESSWIRE|Newsfile Corp\.?)\)\s*[—–-]{1,2}\s*",
    # EN dated datelines without a wire name: "HALIFAX, Nova Scotia, March 05, 2026 — "
    r"^[^\n]{0,120}?\b" + _MONTH + r" \d{1,2}, \d{4}\s*(?:\([^)]*\))?\s*[—–-]{1,2}\s*",
    # ID outlet datelines: "KOMPAS.com - ", "JAKARTA, KOMPAS.com - ", "TEMPO.CO, Jakarta - "
    r"^\s*(?:Oleh:[^\n]{0,80}?\s)?(?:[A-Z][A-Za-z]+,\s*)?(?:KOMPAS\.com|Kompas\.com|"
    r"TEMPO\.CO(?:,\s*[A-Za-z ]+)?|VIVA(?:\.co\.id)?(?:\s*[A-Za-z]+)?|SINDOnews(?:\.com)?|"
    r"detik(?:com|Food|Inet|Finance)?|CNN Indonesia)\s*[—–:-]{1,2}\s*",
    # City datelines: "JAKARTA - ", "MAKKAH - ", "Jakarta - "
    r"^\s*[A-Z][A-Za-z]{2,}(?:\s[A-Z][A-Za-z]+)?\s*[—–-]{1,2}\s+",
]
_ANYWHERE_CUES = [
    r"\b[A-Z][A-Za-z .,'-]{1,40}\((?:AP|Reuters|AFP|CP)\)\s*[—–-]{1,2}\s*",
    r"\((?:GLOBE NEWSWIRE|GLOBENEWSWIRE)\)",
    r"\bGLOBE NEWSWIRE\b|\bGlobeNewswire\b",
    r"\[Gambas:[^\]]*\]",
    r"https?://\S+",
    r"\bSCROLL TO CONTINUE WITH CONTENT\b|\bADVERTISEMENT\b",
    r"(?i)\bbaca juga\s*:[^\n]*",
]


def mask_source_cues(text: str) -> str:
    t = str(text)
    for _ in range(3):  # headers can stack, e.g. legal notice then dateline
        before = t
        t = re.sub(r"^[\s,.;:—–-]+", "", t)
        for pat in _LEADING_CUES:
            t = re.sub(pat, "", t, count=1)
        if t == before:
            break
    for pat in _ANYWHERE_CUES:
        t = re.sub(pat, " ", t)
    t = re.sub(r"^[\s,.;:—–-]+", "", t)  # e.g. Tempo articles that start with ", "
    return re.sub(r"[ \t]+", " ", t).strip()


# ---------------------------------------------------------------- IO

def read_jsonl(path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def env_versions() -> Dict[str, str]:
    """Package versions for the reproducibility record (R1-C12)."""
    import importlib
    import platform
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "peft", "trl", "unsloth", "datasets",
                "sentence_transformers", "numpy", "scikit-learn", "scipy"):
        try:
            m = importlib.import_module(mod.replace("scikit-learn", "sklearn"))
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return out
