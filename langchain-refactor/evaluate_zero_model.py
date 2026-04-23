"""
Scenario 0: Zero-Shot Model Evaluation (No Fine-tuning, No RAG, No Domain Prompt)
==================================================================================
Mirror of evaluate_model.py — same load_test_set(), same metrics pipeline,
same LLM-as-a-Judge, same output format — but loads models via Unsloth
(off-the-shelf, no LoRA, no fine-tuned weights) and uses a minimal binary prompt.

REQUIRED: unsloth (optimized for inference)

Designed to establish a zero-shot baseline comparable to Hu et al. (AAAI 2024)
who showed GPT-3.5 underperforms fine-tuned BERT across all prompting strategies.

Usage:
    python evaluate_zero_model.py \
        --model google/gemma-3-12b-it \
        --dataset ../data/llm_dataset_12k_refined_full.json \
        --use-judge \
        --judge-provider openrouter \
        --judge-model openai/gpt-4o-mini \
        --api-key sk-or-v1-... \
        --num-sample 200

Recommended models (run each separately):
    google/gemma-3-12b-it
    Qwen/Qwen3-8B
    google/gemma-2-9b-it
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B
"""

import os
import sys
import json
import re
import shutil
import random
import time
import subprocess as _sp
import sysconfig as _sc
from pathlib import Path

# Force UTF-8 for stdout/stderr in supercomputer environments
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Fix for Triton compiler error: monkey-patch triton to use ziglang if no system compiler
# ─────────────────────────────────────────────────────────────────────────────

_ZIG_BAD_FLAGS = {"-Wno-psabi", "-Wno-format-truncation",
                  "-mno-avx512f", "-fno-semantic-interposition"}

def _find_cc():
    """Find first available C compiler, trying ziglang as last resort."""
    for _c in ["gcc", "clang", "cc", "g++",
               "/usr/bin/gcc", "/usr/bin/clang",
               _sc.get_config_var("CC")]:
        if _c and shutil.which(str(_c).split()[0]):
            return str(_c).split()[0]
    # Try ziglang (pip install ziglang)
    try:
        import ziglang as _zig
        _zig_bin = os.path.join(os.path.dirname(_zig.__file__), "zig")
        if os.path.exists(_zig_bin):
            _wrapper = "/tmp/_zig_cc"
            with open(_wrapper, "w") as _f:
                _f.write(f'#!/bin/sh\nexec "{_zig_bin}" cc "$@"\n')
            os.chmod(_wrapper, 0o755)
            print(f"✅ Triton will use ziglang C compiler via {_wrapper}")
            return _wrapper
    except ImportError:
        pass
    return None

_CC = _find_cc()
if _CC:
    os.environ["CC"] = _CC

# Fix CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_orig_check_call = _sp.check_call

def _find_python_h_dir():
    """Return the directory containing Python.h using sysconfig for accuracy."""
    import sysconfig
    import sys as _sys
    
    curr_v = f"{_sys.version_info.major}.{_sys.version_info.minor}"
    pyver = f"python{curr_v}"
    
    # 1. Ask sysconfig directly
    inc_dir = sysconfig.get_path("include")
    
    candidates = [
        # The specific path found on your supercomputer
        "/drive0-storage/brian/agentic-ai-native-ads/venv/lib/python3.10/site-packages/tensorflow/include/external/local_config_python/python_include",
        inc_dir,
        os.path.join(inc_dir, pyver) if inc_dir else None,
        # venv include
        os.path.join(os.environ.get("VIRTUAL_ENV", ""), "include", pyver),
        os.path.join(os.environ.get("VIRTUAL_ENV", ""), "include"),
        # Multi-arch paths (Ubuntu/Debian)
        f"/usr/include/x86_64-linux-gnu/{pyver}",
        f"/usr/include/{pyver}",
        f"/usr/local/include/{pyver}",
        # UV paths
        os.path.expanduser(f"~/.local/share/uv/python/cpython-{curr_v}.*/include/{pyver}"),
    ]
    
    # Handle globbing for UV paths
    import glob
    expanded_candidates = []
    for c in candidates:
        if c and "*" in c: expanded_candidates.extend(glob.glob(c))
        elif c: expanded_candidates.append(c)
    
    print(f"🔍 Searching for {pyver} headers in {len(expanded_candidates)} locations...")
    for d in expanded_candidates:
        if d and os.path.exists(os.path.join(d, "Python.h")):
            print(f"✅ Found matching Python.h at: {d}")
            return d
            
    print(f"❌ Could not find headers for {pyver}! Triton compilation will likely fail or use wrong version.")
    return None

_PYTHON_H_DIR = _find_python_h_dir()

if _PYTHON_H_DIR:
    print(f"✅ Found Python.h at: {_PYTHON_H_DIR}")
    # Set CPATH globally so subprocesses (Triton workers) find it automatically
    os.environ["CPATH"] = os.environ.get("CPATH", "") + ":" + _PYTHON_H_DIR
else:
    print("⚠️  Python.h not found — triton compilation may fail.")

import re as _re
_PYTHON_INC_RE = _re.compile(r"-I.*[/\\]python3\.\d+/?$")

def _filtered_check_call(cmd, *args, **kwargs):
    if isinstance(cmd, list) and _CC and "zig" in _CC:
        new_cmd = []
        _py_replaced = False
        for x in cmd:
            if x in _ZIG_BAD_FLAGS: continue
            if _PYTHON_H_DIR and _PYTHON_INC_RE.match(x):
                if not _py_replaced:
                    new_cmd.append(f"-I{_PYTHON_H_DIR}")
                    _py_replaced = True
            else:
                new_cmd.append(x)
        if _PYTHON_H_DIR and f"-I{_PYTHON_H_DIR}" not in new_cmd:
            new_cmd.append(f"-I{_PYTHON_H_DIR}")
        cmd = new_cmd
    return _orig_check_call(cmd, *args, **kwargs)

_sp.check_call = _filtered_check_call
print(f"✅ Patched subprocess.check_call (CC={_CC or 'not found'}, CPATH={os.environ.get('CPATH')})")

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT UNSLOTH FIRST (as requested by UserWarning)
# ─────────────────────────────────────────────────────────────────────────────
from unsloth import FastLanguageModel
import torch
from transformers import AutoTokenizer

# HAS_UNSLOTH is now assumed True since we're using Unsloth exclusively
HAS_UNSLOTH = True

import argparse

# ─────────────────────────────────────────────────────────────────────────────
# Optional dependencies — same guard pattern as evaluate_model.py
# ─────────────────────────────────────────────────────────────────────────────

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn/matplotlib not installed. pip install scikit-learn matplotlib seaborn")

try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("Warning: bert-score not installed. pip install bert-score")

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt',   quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"Note: Could not download NLTK data: {e}")
    HAS_NLP_METRICS = True
except ImportError:
    HAS_NLP_METRICS = False
    print("Warning: nltk not installed. pip install nltk")



# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot prompts — improved with richer native ads indicators
# Mirrors Hu et al. (AAAI 2024) zero-shot setup for fair comparison
# ─────────────────────────────────────────────────────────────────────────────

ZERO_SHOT_PROMPT_ID = """Kamu adalah editor berita senior yang bertugas menentukan apakah sebuah artikel adalah **native ads** atau **berita murni**.

## Definisi:
- **Native Ads**: Konten berbayar atau konten PR yang dirancang menyerupai berita editorial, namun memiliki tujuan komersial tersembunyi atau eksplisit. Termasuk siaran pers perusahaan, laporan keuangan korporat, promosi produk/layanan, dan artikel "advertorial".
- **Berita Murni**: Laporan jurnalistik yang independen, netral, dan tidak memiliki agenda komersial.

## Indikator Native Ads (perhatikan tanda-tanda ini):
1. Artikel berasal dari atau tentang satu perusahaan/brand secara dominan
2. Ada kutipan CEO/direktur/juru bicara perusahaan yang memuji produk/kinerja perusahaan
3. Format siaran pers: kota, tanggal, nama distributor berita (ANTARA, Globe Newswire, PR Newswire, Business Wire, Bisnis.com, VIVA, dll)
4. Menyebut angka keuangan perusahaan (pendapatan, laba, pertumbuhan) dari sudut pandang positif perusahaan
5. Ada ajakan bertindak (call-to-action): kunjungi website, daftar sekarang, hubungi kami
6. Artikel membahas fitur/keunggulan produk atau program perusahaan
7. Pemerintah/BUMN mempromosikan program atau layanan tertentu secara tidak netral

## Indikator Berita Murni:
1. Meliput peristiwa dengan sudut pandang beragam (berbagai narasumber)
2. Tidak ada promosi eksplisit produk/brand tertentu
3. Melaporkan fakta negatif/kritik terhadap perusahaan atau pemerintah
4. Topik: bencana, kriminalitas, politik, olahraga, sains, budaya — tanpa agenda komersial

## Artikel:
{article}

## Instruksi:
Analisis artikel di atas. Tentukan label berdasarkan konten dominan artikel.
Jika ragu antara keduanya, pilih native ads jika ada elemen promosi korporat, bahkan jika tersamar sebagai berita.

Jawab dengan TEPAT salah satu label berikut (tanpa penjelasan tambahan):
**native ads** atau **berita murni**

Label:"""

ZERO_SHOT_PROMPT_EN = """You are a senior news editor tasked with determining whether an article is **native advertising** or **pure news**.

## Definitions:
- **Native Ads**: Paid or PR content designed to look like editorial news while serving a commercial purpose. Includes corporate press releases, financial results, product/service promotions, and advertorials.
- **Pure News**: Independent, neutral journalistic reporting with no commercial agenda.

## Native Ads Indicators (watch for these signals):
1. Article is primarily about or originates from a single company/brand
2. Contains CEO/executive/spokesperson quotes praising company performance or products
3. Press release format: city, date, wire service name (Globe Newswire, PR Newswire, Business Wire, AP Newswire)
4. Reports company financials (revenue, profit, growth) from a positive corporate angle
5. Contains call-to-action: visit website, register now, contact us, learn more
6. Discusses product features, advantages, or company programs
7. Government promoting specific programs or services non-neutrally

## Pure News Indicators:
1. Reports events with multiple perspectives from diverse sources
2. No explicit promotion of any specific product or brand
3. Reports negative facts or criticism of companies/governments
4. Topics: natural disasters, crime, politics, sports, science, culture — without commercial agenda

## Article:
{article}

## Instructions:
Analyze the article above. Determine the label based on the article's dominant content.
If uncertain between the two, choose Native Ads if there are corporate promotional elements, even if disguised as news.

Answer with EXACTLY one of the following labels (no additional explanation):
**Native Ads** or **Pure News**

Label:"""


def build_zero_shot_prompt(article_text: str, lang: str) -> str:
    """Build zero-shot prompt with richer native ads indicators — no domain prompt, no RAG examples."""
    article_text = article_text.strip()
    if lang == "en":
        return ZERO_SHOT_PROMPT_EN.format(article=article_text)
    return ZERO_SHOT_PROMPT_ID.format(article=article_text)


# ─────────────────────────────────────────────────────────────────────────────
# Label parsing — handles mixed-language model outputs
# ─────────────────────────────────────────────────────────────────────────────

def parse_zero_shot_label(raw_output: str, lang: str, model_name: str = "") -> str:
    """
    Truly dynamic label parsing that adapts to different model 'personalities':
    - DeepSeek-R1: Handles <think> tags, GPT2 BPE artifacts (Ġ/Ċ), output "editorial".
    - Qwen3: Handles <think> cut-off (reasoning models that didn't finish thinking).
    - Qwen/Gemma: Handles concatenated words (PureNews) and instruction repetition.
    """
    # ── Fix: Clean GPT2 BPE whitespace artifacts from DeepSeek-R1 / LLaMA tokenizers ──
    # U+0120 (Ġ) represents a space prefix in GPT2-style BPE tokenization
    # U+010A (Ċ) represents a newline in GPT2-style BPE tokenization
    raw_output = raw_output.replace('\u0120', ' ').replace('\u010a', '\n')

    text = raw_output.lower()

    # Clean up common chat template tokens (extended for DeepSeek-R1's <｜end of sentence｜>)
    for token in ["<|im_start|>", "<|im_end|>", "</s>", "<s>", "<pad>",
                  "<｜end of sentence｜>", "<|endoftext|>", "[end]", "[/end]"]:
        text = text.replace(token, "")

    # 1. Handle Reasoning Models (DeepSeek-R1 / Qwen3 use <think> and </think>)
    if "</think>" in text:
        # Take everything AFTER the last </think> tag
        post_think = text.split("</think>")[-1].strip()
        # Sometimes the model outputs nothing after </think> — fall back to last part of reasoning
        text = post_think if post_think else text.split("</think>")[0][-400:]
    elif "</thought>" in text:
        text = text.split("</thought>")[-1]
    elif "<think>" in text:  # Has start tag but no end tag (cut off during generation)
        # Check if the answer is BEFORE the think tag (sometimes happens)
        before_think = text.split("<think>")[0].strip()
        if before_think:
            text = before_think
        else:
            # ── Cut-off thinking recovery ──
            # Model ran out of tokens mid-reasoning.
            # Take the last 3 non-empty lines (most likely to contain conclusion)
            inside_think = text.split("<think>", 1)[1]
            lines = [l.strip() for l in inside_think.strip().split('\n') if l.strip()]
            text = ' '.join(lines[-3:]) if len(lines) >= 3 else (lines[-1] if lines else inside_think[-400:])

    text = text.strip()

    # 2. Strict Pattern Matching (Priority 1)
    # Models often output "Label: native ads" or "**Label:** Pure News"
    strict_match = re.search(
        r'(?:label|jawaban|kategori|prediction|answer|kesimpulan|conclusion|result)'
        r'\s*(?:is|nya|:|=|-)?\s*\*?\[?[\'"]?'
        r'(native ads?|native advertising|iklan native|berita murni|pure news|editorial)',
        text
    )
    if strict_match:
        matched_val = strict_match.group(1).lower()
        if "native" in matched_val or "iklan" in matched_val:
            return "native ads" if lang == "id" else "Native Ads"
        # "editorial" without "native" = pure news (DeepSeek pattern)
        if "murni" in matched_val or "pure" in matched_val or matched_val == "editorial":
            return "berita murni" if lang == "id" else "Pure News"

    # 2b. Match bold markdown format from new prompt: **native ads** or **berita murni**
    bold_match = re.search(r'\*\*(native ads?|native advertising|iklan native|berita murni|pure news)\*\*', text)
    if bold_match:
        matched_val = bold_match.group(1).lower()
        if "native" in matched_val or "iklan" in matched_val:
            return "native ads" if lang == "id" else "Native Ads"
        if "murni" in matched_val or "pure" in matched_val:
            return "berita murni" if lang == "id" else "Pure News"

    # 2c. Direct sentence conclusion patterns (DeepSeek-R1 pattern)
    # e.g. "the article is an editorial", "artikel ini adalah berita murni"
    conclusion_pure = re.search(
        r'(?:article|artikel|konten|teks)\s+(?:ini\s+)?(?:is|merupakan|adalah|termasuk)\s+'
        r'(?:an?\s+)?(?:editorial|pure news|berita murni|berita|news article)',
        text
    )
    if conclusion_pure:
        return "berita murni" if lang == "id" else "Pure News"

    conclusion_native = re.search(
        r'(?:article|artikel|konten|teks)\s+(?:ini\s+)?(?:is|merupakan|adalah|termasuk)\s+'
        r'(?:an?\s+)?(?:native ad|native advertising|iklan native|sponsored|advertorial)',
        text
    )
    if conclusion_native:
        return "native ads" if lang == "id" else "Native Ads"

    # 3. Aggressive Substring Matching
    # IMPORTANT: Do NOT include wire service names (globe newswire, press release) here —
    # models always discuss these in their reasoning, not as final labels.
    # Only include terms the model would use as a CONCLUSION word.
    native_kws = [
        "native ads", "nativeads", "native advertising", "nativeadvertising",
        "iklan native", "iklannative", "advertorial", "sponsored content",
        "konten berbayar", "konten pr", "konten promosi",
    ]
    pure_kws = [
        "berita murni", "beritamurni", "pure news", "purenews",
        "legitimate editorial", "jurnalistik", "bukan iklan", "bukaniklan",
        "independent report", "editorial independen",
        # DeepSeek-R1 specific: concludes with just "editorial" or "an editorial"
        " is an editorial", " is editorial", "adalah editorial", "merupakan editorial",
        "artikel editorial", "news article", "news report",
    ]

    # 4. Dynamic Check: Find the LAST occurrence of either keyword set
    # This prevents false positives where the model discusses "native ads" in reasoning but concludes "pure news".
    last_native_idx = max([text.rfind(kw) for kw in native_kws] + [-1])
    last_pure_idx   = max([text.rfind(kw) for kw in pure_kws] + [-1])

    if last_native_idx == -1 and last_pure_idx == -1:
        # 5. Fuzzy Word-level Check (scan from end of text toward beginning)
        words = text.replace("-", " ").replace("_", " ").split()
        for w in reversed(words):
            if w in {"native", "ads", "iklan", "sponsored", "advertorial"}:
                return "native ads" if lang == "id" else "Native Ads"
            # "berita" removed — too ambiguous (appears in all model reasoning text constantly)
            if w in {"murni", "editorial"}:
                return "berita murni" if lang == "id" else "Pure News"
        # Last resort: "pure" or "news" as standalone word usually = pure news conclusion
        if re.search(r'\bpure\b|\bnews\b', text):
            return "berita murni" if lang == "id" else "Pure News"
        return "Unknown"

    if last_native_idx > last_pure_idx:
        return "native ads" if lang == "id" else "Native Ads"
    else:
        return "berita murni" if lang == "id" else "Pure News"


def normalize_gt_label(raw_label: str) -> str:
    """Normalize ground truth label to lowercase for consistent comparison."""
    label = raw_label.lower().strip()
    if any(k in label for k in ["native", "iklan"]):
        return "native ads"
    if any(k in label for k in ["pure", "murni"]):
        return "berita murni"
    return label


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading — IDENTICAL to evaluate_model.py
# ─────────────────────────────────────────────────────────────────────────────

def load_test_set(dataset_path: str, num_samples: int = 200, lang_filter: str = None) -> List[Dict]:
    """
    Load test samples from dataset.
    Identical logic to evaluate_model.py: seed-42 shuffle, proportional lang sampling.
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if lang_filter:
        data = [item for item in data if item.get('lang') == lang_filter.lower()]
        print(f"🌍 Filtered by language: {lang_filter} ({len(data)} samples found)")
        random.seed(42)
        random.shuffle(data)
        test_data = data[:num_samples]
    else:
        # Proportional sampling between languages
        lang_groups = defaultdict(list)
        for item in data:
            lang = item.get('lang', 'id').lower()
            lang_groups[lang].append(item)

        total_available = len(data)
        test_data = []

        print(f"🌍 No language filter. Sampling proportionally from {len(lang_groups)} language(s)...")

        random.seed(42)

        for lang, items in lang_groups.items():
            proportion = len(items) / total_available
            lang_samples = int(num_samples * proportion)
            random.shuffle(items)
            test_data.extend(items[:lang_samples])
            print(f"   - {lang}: {lang_samples} samples")

        # Fill rounding gap from largest group
        if len(test_data) < num_samples:
            diff = num_samples - len(test_data)
            largest_lang = max(lang_groups, key=lambda k: len(lang_groups[k]))
            current_ids = {item['id'] for item in test_data}
            remaining = [item for item in lang_groups[largest_lang] if item['id'] not in current_ids]
            test_data.extend(remaining[:diff])

    # Final shuffle — same seed as evaluate_model.py
    random.seed(42)
    random.shuffle(test_data)

    from collections import Counter
    dist = Counter()
    for item in test_data:
        try:
            output = json.loads(item['output'])
            dist[output['label']] += 1
        except Exception:
            pass
    print(f"📊 Test set distribution: {dict(dist)}")

    return test_data


# ─────────────────────────────────────────────────────────────────────────────
# Model loading via Unsloth
# ─────────────────────────────────────────────────────────────────────────────

def load_model_unsloth(model_name: str, max_seq_length: int = 2048, gpu_id: int = 0):
    """
    Load off-the-shelf model via Unsloth (no LoRA, no fine-tuned weights).
    """
    print(f"🦥 Loading via Unsloth: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,          # Auto-detect (bfloat16 on Ampere+, float16 otherwise)
        load_in_4bit=True,   # 4-bit quant — same memory footprint as fine-tuned eval
    )
    FastLanguageModel.for_inference(model)
    print(f"✅ Unsloth model loaded (4-bit, inference mode)")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def zero_shot_classify(
    model,
    tokenizer,
    article_text: str,
    lang: str,
    max_new_tokens: int = 64,
    max_article_chars: int = 6000,
    model_name: str = "",
    max_seq_length: int = 8192,
) -> Dict:
    """
    Run zero-shot classification on a single article.
    Returns dict with label, confidence (placeholder 1.0), reasoning (raw output).
    """
    # Truncate very long articles to fit context window
    if len(article_text) > max_article_chars:
        article_text = article_text[:max_article_chars] + " [...]"

    prompt = build_zero_shot_prompt(article_text, lang)

    # Use chat template if available (crucial for models like DeepSeek-R1, Qwen-IT, Gemma-IT)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Ensure we leave enough space for max_new_tokens
    prompt_max_length = max(512, max_seq_length - max_new_tokens - 100)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_max_length,
        padding=False,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    latency = time.perf_counter() - t0

    # Decode only newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    # skip_special_tokens=False so we can parse <think> tags from reasoning models.
    # We clean special tokens manually in parse_zero_shot_label.
    raw_text = tokenizer.decode(new_ids, skip_special_tokens=False).strip()

    label = parse_zero_shot_label(raw_text, lang, model_name=model_name)

    return {
        "label":     label,
        "confidence": 1.0,
        "reasoning": raw_text,
        "metadata": {
            "raw_response": raw_text,
            "latency_s": round(latency, 3),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop — mirrors evaluate_model.py structure
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_zero_shot(model_name: str, test_data: List[Dict], **kwargs) -> Dict:
    """Evaluate zero-shot model on test set. Mirrors evaluate_model() structure."""

    print(f"Loading model: {model_name}")
    model, tokenizer = load_model_unsloth(
        model_name,
        max_seq_length=kwargs.get('max_seq_length', 2048),
        gpu_id=kwargs.get('gpu_id', 0),
    )

    results = {
        'correct': 0,
        'total': 0,
        'by_class': defaultdict(lambda: {'correct': 0, 'total': 0, 'predictions': []}),
        'confusion_matrix': defaultdict(int),
        'predictions': [],
        'ground_truth_labels': [],
        'predicted_labels': [],
        'reasoning_texts': {'ground_truth': [], 'predicted': []},
        'perplexities': [],   # Empty for zero-shot (no perplexity without fine-tuned logits)
        'latencies': [],
    }

    print(f"\nEvaluating {len(test_data)} samples (zero-shot, no RAG, no domain prompt)...")

    for i, sample in enumerate(test_data):
        try:
            gt_data     = json.loads(sample['output'])
            gt_label_raw = gt_data['label']
            gt_label    = normalize_gt_label(gt_label_raw)
            gt_reasoning = gt_data.get('reasoning', '')
        except Exception:
            continue

        lang = sample.get('lang', 'id').lower()

        try:
            eff_max_tokens = kwargs.get('max_new_tokens', 512)
            # Auto-bump max_new_tokens based on model family:
            model_lower = model_name.lower()
            if "r1" in model_lower or "deepseek" in model_lower:
                # R1 distill models: long thinking chains
                if eff_max_tokens < 3000:
                    eff_max_tokens = 3000
            elif "qwen3" in model_lower or "qwen-3" in model_lower:
                # Qwen3 thinking: moderate reasoning
                if eff_max_tokens < 2048:
                    eff_max_tokens = 2048
            elif "gemma" in model_lower or "llama" in model_lower:
                # Non-reasoning models: 256 is enough for a label + brief explanation
                if eff_max_tokens > 512:
                    eff_max_tokens = 256

            pred = zero_shot_classify(
                model, tokenizer,
                article_text=sample['input'],
                lang=lang,
                max_new_tokens=eff_max_tokens,
                model_name=model_name,
                max_seq_length=kwargs.get('max_seq_length', 8192)
            )
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

        pred_label    = normalize_gt_label(pred['label']) if pred['label'] != "Unknown" else "Unknown"
        pred_reasoning = str(pred.get('reasoning', ''))
        latency        = pred.get('metadata', {}).get('latency_s', 0)
        results['latencies'].append(latency)

        # String-coerce gt_reasoning
        if isinstance(gt_reasoning, list):
            gt_reasoning = " ".join(str(x) for x in gt_reasoning)
        else:
            gt_reasoning = str(gt_reasoning)

        results['total'] += 1
        results['by_class'][gt_label]['total'] += 1
        results['ground_truth_labels'].append(gt_label)
        results['predicted_labels'].append(pred_label)
        results['reasoning_texts']['ground_truth'].append(gt_reasoning)
        results['reasoning_texts']['predicted'].append(pred_reasoning)

        is_correct = (pred_label == gt_label)
        if is_correct:
            results['correct'] += 1
            results['by_class'][gt_label]['correct'] += 1

        results['confusion_matrix'][f"{gt_label} -> {pred_label}"] += 1
        results['by_class'][gt_label]['predictions'].append({
            'input':      sample['input'][:200],
            'predicted':  pred_label,
            'confidence': pred.get('confidence', 1.0),
            'correct':    is_correct,
        })
        results['predictions'].append({
            'input':        sample['input'][:200],
            'ground_truth': gt_label,
            'prediction':   pred_label,
            'confidence':   pred.get('confidence', 1.0),
            'gt_reasoning': gt_reasoning,
            'pred_reasoning': pred_reasoning,
            'correct':      is_correct,
            'lang':         lang,
            'latency_s':    latency,
        })

        print(f"[{i+1}/{len(test_data)}] {gt_label} -> {pred_label}  |  {pred.get('metadata',{}).get('raw_response','')[:60]}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Metrics — copied verbatim from evaluate_model.py
# ─────────────────────────────────────────────────────────────────────────────

def compute_bertscore(results: Dict) -> Dict:
    """Compute BERTScore for reasoning quality."""
    if not HAS_BERTSCORE:
        return {}

    print("\n📊 Computing BERTScore for reasoning quality (CPU mode)...")

    gt_reasoning   = results['reasoning_texts']['ground_truth']
    pred_reasoning = results['reasoning_texts']['predicted']

    valid_pairs = [(gt, pred) for gt, pred in zip(gt_reasoning, pred_reasoning) if gt and pred]
    if not valid_pairs:
        return {}

    gt_texts, pred_texts = zip(*valid_pairs)
    gt_texts   = [str(t) for t in gt_texts]
    pred_texts = [str(t) for t in pred_texts]

    try:
        P, R, F1 = bert_score(
            pred_texts, gt_texts,
            model_type='bert-base-multilingual-cased',
            verbose=False, device='cpu'
        )
        return {
            'precision': float(P.mean()),
            'recall':    float(R.mean()),
            'f1':        float(F1.mean()),
        }
    except Exception as e:
        print(f"⚠️  BERTScore failed: {e}")
        return {}


def compute_nlp_metrics(results: Dict) -> Dict:
    """Compute BLEU and METEOR for reasoning quality."""
    if not HAS_NLP_METRICS:
        return {}

    print("\n📝 Computing BLEU and METEOR metrics...")

    gt_reasoning   = results['reasoning_texts']['ground_truth']
    pred_reasoning = results['reasoning_texts']['predicted']

    bleu_scores   = []
    meteor_scores = []
    smoothie      = SmoothingFunction().method1
    processed     = 0

    for gt, pred in zip(gt_reasoning, pred_reasoning):
        if not gt or not pred:
            continue
        processed += 1
        gt_tokens   = nltk.word_tokenize(str(gt).lower())
        pred_tokens = nltk.word_tokenize(str(pred).lower())

        bleu_scores.append(sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothie))
        try:
            meteor_scores.append(meteor_score([gt_tokens], pred_tokens))
        except Exception:
            pass

    print(f"   (Processed {processed} samples for NLP metrics)")

    return {
        'bleu':   np.mean(bleu_scores)   if bleu_scores   else 0,
        'meteor': np.mean(meteor_scores) if meteor_scores else 0,
    }


def llm_as_judge(
    results: Dict,
    api_key: str = None,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
) -> Dict:
    """
    Use LLM-as-a-Judge to evaluate zero-shot prediction quality.
    Identical prompt and sampling logic to evaluate_model.py.
    """
    if not HAS_OPENAI:
        print("⚠️  Skipping LLM-as-a-Judge: langchain-openai not installed.")
        return {}
    if not api_key:
        print("⚠️  Skipping LLM-as-a-Judge: API Key missing (--api-key).")
        return {}

    print(f"\n🤖 Running LLM-as-a-Judge (Provider: {provider}, Model: {model})...")

    if provider == "openrouter":
        llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            default_headers={
                "HTTP-Referer": "https://github.com/brianrizqi/agentic-ai-native-ads",
                "X-Title": "Agentic Native Ads Evaluation - Scenario 0 Zero-Shot",
            }
        )
    else:
        llm = ChatOpenAI(model=model, openai_api_key=api_key, temperature=0)

    judge_prompt = PromptTemplate(
        template="""You are an expert editor evaluating a Native Ads vs Pure News classifier.
            
ARTICLE CONTENT:
{input_text}

CLASSIFICATION DETAILS:
- Ground Truth: {ground_truth}
- Model Prediction: {prediction}
- Model's Reasoning: {reasoning}

EVALUATION CRITERIA:
1. Label Accuracy: Is the prediction correct according to ground truth?
2. Reasoning Quality: Does the reasoning logically support the label? (Is it objective for news? Is it identifying promotional tone for ads?)
3. Hard Negative Check: If this is news mentioning brands/money, did the model correctly identify it as "Berita Murni" instead of "Native Ads"?

Rate the classification on a scale of 1-5:
1 = Completely wrong (Wrong label, nonsense reasoning)
2 = Mostly wrong (Wrong label, shallow reasoning)
3 = Partially correct (Correct label, but weak/generic reasoning)
4 = Mostly correct (Correct label, logical reasoning)
5 = Perfect (Correct label, deep and accurate reasoning)

Output JSON:
{{"rating": X, "justification": "short explanation"}}""",
        input_variables=["input_text", "ground_truth", "prediction", "reasoning"],
    )

    all_predictions = results['predictions']
    incorrect  = [p for p in all_predictions if not p.get('correct')]
    correct    = [p for p in all_predictions if p.get('correct')]
    sample_size = min(100, len(all_predictions))
    num_incorrect = int(sample_size * 0.8)
    judge_samples = (incorrect[:num_incorrect] + correct)[:sample_size]

    ratings = []
    details = []

    for i, pred in enumerate(judge_samples):
        print(f"   Judging sample {i+1}/{len(judge_samples)}...", end='\r')
        try:
            response = llm.invoke(
                judge_prompt.format(
                    input_text=pred['input'][:2000],
                    ground_truth=pred['ground_truth'],
                    prediction=pred['prediction'],
                    reasoning=pred.get('pred_reasoning', ''),
                )
            )
            judge_result = json.loads(response.content)
            ratings.append(judge_result['rating'])
            details.append({
                'input_preview':      pred['input'][:100] + "...",
                'gt':                 pred['ground_truth'],
                'pred':               pred['prediction'],
                'judge_rating':       judge_result['rating'],
                'judge_justification': judge_result['justification'],
            })
        except Exception as e:
            print(f"\n   Judge error on sample {i}: {e}")
            continue

    print(f"\n✅ Finished judging {len(ratings)} samples.")

    return {
        'average_rating':     np.mean(ratings) if ratings else 0,
        'num_samples_judged': len(ratings),
        'judge_details':      details,
    }


def export_reasoning_examples(
    results: Dict,
    judge_results: Dict,
    output_path: str,
    n_per_category: int = 3,
) -> None:
    """
    Export qualitative reasoning examples — identical to evaluate_model.py.
    Grouped into good (4-5), borderline (3), bad (1-2).
    """
    pred_lookup = {p['input'][:100]: p for p in results.get('predictions', [])}
    judge_details = judge_results.get('judge_details', [])

    if not judge_details:
        print("\n⚠️  No judge details — run with --use-judge first.")
        return

    good, borderline, bad = [], [], []
    for detail in judge_details:
        score = detail.get('judge_rating', 0)
        key   = detail.get('input_preview', '')[:100]
        full  = pred_lookup.get(key, {})
        entry = {
            "judge_score":              score,
            "judge_justification":      detail.get('judge_justification', ''),
            "ground_truth_label":       detail.get('gt', ''),
            "predicted_label":          detail.get('pred', ''),
            "correct":                  detail.get('gt', '') == detail.get('pred', ''),
            "article_preview":          full.get('input', detail.get('input_preview', ''))[:400],
            "ground_truth_reasoning":   full.get('gt_reasoning', '')[:500],
            "model_reasoning":          full.get('pred_reasoning', '')[:500],
            "confidence":               full.get('confidence', None),
        }
        if score >= 4:   good.append(entry)
        elif score == 3: borderline.append(entry)
        else:            bad.append(entry)

    good       = sorted(good,       key=lambda x: -x['judge_score'])[:n_per_category]
    borderline = sorted(borderline, key=lambda x: -x['judge_score'])[:n_per_category]
    bad        = sorted(bad,        key=lambda x:  x['judge_score'])[:n_per_category]

    output = {
        "_description": (
            "Zero-shot qualitative reasoning examples (Scenario 0). "
            "good=score 4-5, borderline=score 3, bad=score 1-2."
        ),
        "summary": {
            "total_judged":       len(judge_details),
            "good_count":         len([d for d in judge_details if d.get('judge_rating', 0) >= 4]),
            "borderline_count":   len([d for d in judge_details if d.get('judge_rating', 0) == 3]),
            "bad_count":          len([d for d in judge_details if d.get('judge_rating', 0) <= 2]),
            "avg_judge_score":    round(judge_results.get('average_rating', 0), 3),
        },
        "good_examples":       good,
        "borderline_examples": borderline,
        "bad_examples":        bad,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n📋 Qualitative reasoning examples saved to: {output_path}")
    print(f"   good={len(good)}, borderline={len(borderline)}, bad={len(bad)}")


def plot_confusion_matrix(results: Dict, output_path: str = 'confusion_matrix.png'):
    """Plot and save confusion matrix — identical to evaluate_model.py."""
    if not HAS_SKLEARN:
        return None

    print("\n📈 Generating confusion matrix...")

    labels = ['berita murni', 'native ads']
    cm = confusion_matrix(
        results['ground_truth_labels'],
        results['predicted_labels'],
        labels=labels,
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Reds',
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Count'},
    )
    plt.title('Confusion Matrix — Zero-Shot Native Ads Detection (Scenario 0)',
              fontsize=14, fontweight='bold')
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Predicted',    fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to: {output_path}")
    return cm


def print_results(
    results: Dict,
    bertscore_results: Dict = None,
    judge_results: Dict     = None,
    cm                      = None,
):
    """Print comprehensive evaluation results — mirrors evaluate_model.py."""

    print("\n" + "=" * 80)
    print("ZERO-SHOT EVALUATION RESULTS  (Scenario 0)")
    print("=" * 80)

    accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy: {accuracy*100:.2f}% ({results['correct']}/{results['total']})")

    if HAS_SKLEARN and results['ground_truth_labels']:
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            results['ground_truth_labels'],
            results['predicted_labels'],
            average='weighted',
            zero_division=0,
        )
        print(f"   Precision (weighted): {precision*100:.2f}%")
        print(f"   Recall    (weighted): {recall*100:.2f}%")
        print(f"   F1-Score  (weighted): {f1*100:.2f}%")
        try:
            kappa = cohen_kappa_score(results['ground_truth_labels'], results['predicted_labels'])
            print(f"   Cohen's Kappa: {kappa:.3f}")
        except Exception:
            pass

    # Per-class
    print(f"\n📈 Per-Class Performance:")
    for label, stats in results['by_class'].items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"   {label}:")
        print(f"      Accuracy: {acc*100:.2f}% ({stats['correct']}/{stats['total']})")
        if HAS_SKLEARN:
            class_preds = [p for p in results['predictions'] if p['ground_truth'] == label]
            if class_preds:
                tp = sum(1 for p in class_preds if p['correct'])
                fp = sum(1 for p in results['predictions']
                         if p['prediction'] == label and not p['correct'])
                fn = len(class_preds) - tp
                pr = tp / (tp + fp) if (tp + fp) > 0 else 0
                re = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * pr * re / (pr + re) if (pr + re) > 0 else 0
                print(f"      Precision: {pr*100:.2f}%")
                print(f"      Recall:    {re*100:.2f}%")
                print(f"      F1-Score:  {f1*100:.2f}%")

    # Unknown predictions (model refused / garbled output)
    unknowns = sum(1 for p in results['predictions'] if p['prediction'] == 'Unknown')
    if unknowns:
        print(f"\n⚠️  Unknown predictions (parse failure): {unknowns}/{results['total']}")

    # Average latency
    if results.get('latencies'):
        print(f"\n⏱️  Avg latency per sample: {np.mean(results['latencies']):.2f}s")

    if results.get('nlp_metrics'):
        m = results['nlp_metrics']
        print(f"\n📝 NLP Metrics (Reasoning Quality — vs dataset reasoning):")
        print(f"   BLEU:   {m.get('bleu', 0):.4f}")
        print(f"   METEOR: {m.get('meteor', 0):.4f}")

    if bertscore_results:
        print(f"\n🎯 BERTScore (Reasoning Quality):")
        print(f"   Precision: {bertscore_results['precision']:.3f}")
        print(f"   Recall:    {bertscore_results['recall']:.3f}")
        print(f"   F1:        {bertscore_results['f1']:.3f}")

    if judge_results and judge_results.get('num_samples_judged', 0) > 0:
        print(f"\n🤖 LLM-as-a-Judge:")
        print(f"   Average Rating:  {judge_results['average_rating']:.2f}/5.0")
        print(f"   Samples Judged:  {judge_results['num_samples_judged']}")

    if cm is not None:
        print(f"\n🔀 Confusion Matrix:")
        print(f"   True Negatives  (Berita Murni → Berita Murni): {cm[0][0]}")
        print(f"   False Positives (Berita Murni → Native Ads):   {cm[0][1]}")
        print(f"   False Negatives (Native Ads   → Berita Murni): {cm[1][0]}")
        print(f"   True Positives  (Native Ads   → Native Ads):   {cm[1][1]}")

    print(f"\n❌ Sample Errors (First 5):")
    errors = [p for p in results['predictions'] if not p['correct']]
    for j, error in enumerate(errors[:5]):
        print(f"\n   Error {j+1}:")
        print(f"   Input:     {error['input']}...")
        print(f"   GT: {error['ground_truth']},  Pred: {error['prediction']}")
        print(f"   Output: {error['pred_reasoning'][:120]}...")


# ─────────────────────────────────────────────────────────────────────────────
# CLI — mirrors evaluate_model.py argument names exactly
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Scenario 0: Zero-Shot Native Ads Detection Baseline'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model ID or local path (off-the-shelf, no fine-tuning)')
    parser.add_argument('--dataset', type=str,
                        default='../data/llm_dataset_12k_refined_full.json',
                        help='Dataset path (same as evaluate_model.py)')
    parser.add_argument('--num-samples', '--num-sample', type=int, default=200,
                        help='Number of test samples (default: 200)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results_scenario0',
                        help='Output directory for results')
    parser.add_argument('--use-judge', action='store_true',
                        help='Use LLM-as-a-Judge (same as evaluate_model.py)')
    parser.add_argument('--judge-provider', type=str, default='openai',
                        choices=['openai', 'openrouter'],
                        help='Provider for LLM judge')
    parser.add_argument('--judge-model', type=str, default='gpt-4o-mini',
                        help='Model for LLM judge (e.g. openai/gpt-4o-mini for openrouter)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key for LLM judge')
    parser.add_argument('--lang', type=str, default=None,
                        help='Filter by language (en / id)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID (default: 0)')
    parser.add_argument('--max-new-tokens', type=int, default=512,
                        help='Max tokens to generate per sample (default: 512)')
    parser.add_argument('--max-seq-length', type=int, default=8192,
                        help='Max sequence length for Unsloth (default: 8192)')
    args = parser.parse_args()

    # ── Output naming — same convention as evaluate_model.py ──────────────
    model_path  = Path(args.model)
    model_name  = model_path.name.replace('_merged_16bit', '').replace('-', '_')
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'eval_zero_{model_name}_{timestamp}'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Scenario 0 — Zero-Shot Model Evaluation")
    print(f"{'='*80}")
    print(f"Model:    {args.model}")
    print(f"Dataset:  {args.dataset}")
    print(f"Samples:  {args.num_samples}")
    print(f"Mode:     zero-shot | no fine-tuning | no RAG | no domain prompt")
    print(f"Output:   {output_dir / output_filename}")
    print(f"{'='*80}\n")

    # ── Load test set (identical to evaluate_model.py) ────────────────────
    test_data = load_test_set(args.dataset, args.num_samples, args.lang)

    # ── Run evaluation ─────────────────────────────────────────────────────
    results = evaluate_zero_shot(
        args.model,
        test_data,
        gpu_id=args.gpu_id,
        max_new_tokens=args.max_new_tokens,
        max_seq_length=args.max_seq_length,
    )

    # ── Metrics ───────────────────────────────────────────────────────────
    bertscore_results = compute_bertscore(results)
    nlp_results       = compute_nlp_metrics(results)
    results['nlp_metrics'] = nlp_results

    judge_results = {}
    if args.use_judge:
        judge_results = llm_as_judge(
            results, args.api_key, args.judge_provider, args.judge_model
        )
        examples_path = output_dir / f'{output_filename}_reasoning_examples.json'
        export_reasoning_examples(results, judge_results, str(examples_path))

    cm_path = output_dir / f'{output_filename}_confusion_matrix.png'
    cm      = plot_confusion_matrix(results, str(cm_path))

    print_results(results, bertscore_results, judge_results, cm)

    # ── Save JSON + summary — same structure as evaluate_model.py ─────────
    accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0

    output_data = {
        'scenario':   'Scenario 0 — Zero-Shot (no fine-tuning, no RAG, no domain prompt)',
        'model':      args.model,
        'model_name': model_name,
        'timestamp':  timestamp,
        'dataset':    args.dataset,
        'num_samples': args.num_samples,
        'metrics': {
            'accuracy':      accuracy,
            'total_samples': results['total'],
            'correct':       results['correct'],
        },
        'bertscore':     bertscore_results,
        'nlp_metrics':   nlp_results,
        'avg_perplexity': 0,   # Not computed for zero-shot
        'avg_latency_s':  float(np.mean(results['latencies'])) if results['latencies'] else 0,
        'llm_judge':     judge_results,
        'per_class': {
            k: {'accuracy': v['correct'] / v['total']}
            for k, v in results['by_class'].items()
        },
        'predictions': results['predictions'],
    }

    json_path    = output_dir / f'{output_filename}.json'
    summary_path = output_dir / f'{output_filename}_summary.txt'

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"SCENARIO 0 — ZERO-SHOT EVALUATION SUMMARY\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Model:     {args.model}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset:   {args.dataset}\n")
        f.write(f"Samples:   {args.num_samples}\n")
        f.write(f"Mode:      zero-shot | no fine-tuning | no RAG | no domain prompt\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"  Correct:  {results['correct']}/{results['total']}\n\n")
        if bertscore_results:
            f.write(f"BERTScore:\n")
            f.write(f"  Precision: {bertscore_results['precision']:.3f}\n")
            f.write(f"  Recall:    {bertscore_results['recall']:.3f}\n")
            f.write(f"  F1:        {bertscore_results['f1']:.3f}\n\n")
        if nlp_results:
            f.write(f"NLP Metrics (Reasoning Quality):\n")
            f.write(f"  BLEU:   {nlp_results['bleu']:.4f}\n")
            f.write(f"  METEOR: {nlp_results['meteor']:.4f}\n\n")
        f.write(f"Per-Class Accuracy:\n")
        for label, stats in output_data['per_class'].items():
            f.write(f"  {label}: {stats['accuracy']*100:.2f}%\n")
        if judge_results and judge_results.get('num_samples_judged', 0) > 0:
            f.write(f"\nLLM-as-a-Judge:\n")
            f.write(f"  Average Rating:  {judge_results['average_rating']:.2f}/5.0\n")
            f.write(f"  Samples Judged:  {judge_results['num_samples_judged']}\n")

    print(f"\n{'='*80}")
    print(f"✅ Results saved to:")
    print(f"  📄 JSON:            {json_path}")
    print(f"  📝 Summary:         {summary_path}")
    print(f"  📊 Confusion Matrix: {cm_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
