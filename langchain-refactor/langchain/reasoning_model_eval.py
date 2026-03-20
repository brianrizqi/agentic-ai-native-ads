"""
Reasoning Model Evaluation Script
===================================
Mengevaluasi dan membandingkan tiga model via OpenRouter:

  [EXPLICIT THINKING] — model menulis <think>...</think> sebelum jawaban
    - Qwen3 8B         : qwen/qwen3-8b
    - DeepSeek-R1 Llama 8B : deepseek/deepseek-r1-distill-llama-8b

  [IMPLICIT REASONING] — model langsung jawab tanpa thinking block
    - Gemma 3 12B      : google/gemma-3-12b-it

Pertanyaan penelitian:
  Apakah explicit chain-of-thought (<think>) meningkatkan kualitas
  klasifikasi native ads dibanding model yang reasoning-nya implisit?

Output:
  - JSON metrics per model (accuracy, F1, BERTScore, BLEU, METEOR)
  - JSON qualitative examples (good / borderline / bad reasoning)
  - PNG: bar chart metrics, radar chart explicit vs implicit
  - CSV comparison table

Usage:
  cd langchain-refactor
  python langchain/reasoning_model_eval.py \\
      --dataset ../data/llm_dataset_finetuning_optimized.json \\
      --num-samples 50 \\
      --api-key sk-or-v1-xxx \\
      --output langchain/reasoning_model_results
"""

import json
import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import numpy as np

# ─── Optional imports ─────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("⚠️  matplotlib/seaborn not installed: pip install matplotlib seaborn")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from bert_score import score as bert_score_fn
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("⚠️  bert-score not installed: pip install bert-score")

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    HAS_NLP = True
except ImportError:
    HAS_NLP = False
    print("⚠️  nltk not installed: pip install nltk")

try:
    from langchain_openai import ChatOpenAI
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    print("⚠️  langchain-openai not installed: pip install langchain-openai")

# ─── Model Definitions ────────────────────────────────────────────────────────

MODELS = {
    "Qwen3-8B": {
        "id":       "qwen/qwen3-8b",
        "thinking": True,   # Model keluarkan <think>...</think>
        "family":   "qwen",
        "params":   "8B",
    },
    "DeepSeek-R1-Llama-8B": {
        "id":       "deepseek/deepseek-r1-distill-llama-8b",
        "thinking": True,   # Model keluarkan <think>...</think>
        "family":   "llama",
        "params":   "8B",
    },
    "Gemma3-12B": {
        "id":       "google/gemma-3-12b-it",
        "thinking": False,  # Implicit reasoning — langsung jawab
        "family":   "gemma",
        "params":   "12B",
    },
}

# ─── Prompts ──────────────────────────────────────────────────────────────────

# Untuk model thinking (Qwen3, DeepSeek-R1): instruksikan pakai thinking
THINKING_PROMPT = """You are an expert Indonesian news editor.
Classify the following article as "native ads" or "berita murni" (pure news).

DEFINITIONS:
- native ads: content that promotes a product/brand/institution with positive/persuasive tone, single viewpoint.
- berita murni: objective reporting, can be positive/negative/neutral, multiple viewpoints, no promotion agenda.

ARTICLE:
Title: {title}
Content: {content}

Think step by step before concluding.
Output your final answer as JSON (after your thinking):
{{"reasoning": "<brief summary of your analysis>", "label": "<native ads | berita murni>", "confidence": <0.0-1.0>}}"""

# Untuk model non-thinking (Gemma 3 12B): instruksikan reasoning di output
STANDARD_PROMPT = """Kamu adalah editor berita Indonesia yang ahli.
Klasifikasikan artikel berikut sebagai "native ads" atau "berita murni".

DEFINISI:
- Native Ads: konten yang mempromosikan produk/brand dengan nada persuasif, satu sudut pandang.
- Berita Murni: liputan objektif, berbagai sudut pandang, tidak ada agenda promosi.

ARTIKEL:
Judul: {title}
Konten: {content}

Berikan reasoning terlebih dahulu, lalu output JSON:
{{"reasoning": "<analisis singkat>", "label": "<native ads | berita murni>", "confidence": <0.0-1.0>}}"""

# Judge prompt — menilai kualitas proses berpikir (thinking vs non-thinking)
JUDGE_PROMPT = """You are an expert evaluator for native ads classification.

ARTICLE:
{input_text}

GROUND TRUTH: {ground_truth}
MODEL: {model_name} ({thinking_type})
PREDICTION: {prediction}
THINKING PROCESS (if any): {thinking_process}
REASONING SUMMARY: {reasoning}

Evaluate the quality of the model's reasoning/thinking:
1. Did the model identify the correct signals (promotional tone, brand mentions, persuasive language)?
2. Is the thinking process coherent and relevant to the task?
3. For hard cases (news mentioning brands), did the model correctly distinguish native ads from pure news?
4. Is the final label consistent with the reasoning?

Score 1-5:
1 = Wrong label, incoherent reasoning
2 = Wrong label or very shallow reasoning
3 = Correct label, but reasoning is generic/surface-level
4 = Correct label, relevant and clear reasoning
5 = Correct label, deep analysis, correctly handles hard cases

Output JSON:
{{"score": <1-5>, "justification": "<max 2 sentences>", "thinking_quality": "<deep|adequate|shallow|absent>"}}"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_llm(model_id: str, api_key: str, temperature: float = 0.1) -> "ChatOpenAI":
    return ChatOpenAI(
        model=model_id,
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        default_headers={
            "HTTP-Referer": "https://github.com/brianrizqi/agentic-ai-native-ads",
            "X-Title": "Native Ads Reasoning Model Evaluation"
        }
    )


def _call_with_retry(llm, prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    return ""


def _extract_thinking(raw: str) -> Tuple[str, str]:
    """
    Pisahkan <think>...</think> dari sisa response.
    Return: (thinking_process, rest_of_response)
    """
    think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL | re.IGNORECASE)
    if think_match:
        thinking = think_match.group(1).strip()
        rest = raw[think_match.end():].strip()
        return thinking, rest
    return "", raw


def _parse_json(text: str) -> Dict:
    """Parse JSON dari response, toleran terhadap markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Coba ekstrak JSON dari teks panjang
    m = re.search(r'\{[^{}]*"label"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}


def _normalize_label(pred_label: str) -> str:
    pl = pred_label.strip().lower()
    if 'native' in pl:
        return 'native ads'
    if 'murni' in pl or 'berita' in pl or 'pure' in pl or 'news' in pl:
        return 'berita murni'
    return 'unknown'


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_test_samples(dataset_path: str, num_samples: int = 50) -> List[Dict]:
    """Load data test secara proporsional per bahasa dan label."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    import random
    random.seed(42)

    valid = []
    for item in data:
        try:
            out = json.loads(item['output'])
            if out.get('label') and out.get('reasoning') and len(str(out['reasoning'])) > 10:
                valid.append(item)
        except Exception:
            continue

    # Proporsional per bahasa
    lang_groups: Dict[str, List] = defaultdict(list)
    for item in valid:
        lang_groups[item.get('lang', 'id').lower()].append(item)

    test_data = []
    total = len(valid)
    for lang, items in lang_groups.items():
        n = max(1, int(num_samples * len(items) / total))
        random.shuffle(items)
        test_data.extend(items[:n])
        print(f"   [{lang.upper()}] {min(n, len(items))} samples")

    random.shuffle(test_data)
    test_data = test_data[:num_samples]

    from collections import Counter
    dist = Counter(json.loads(i['output'])['label'] for i in test_data
                   if i['output'])
    print(f"   Labels: {dict(dist)}")
    return test_data


# ─── Per-Model Evaluation ─────────────────────────────────────────────────────

def evaluate_model(model_name: str, model_cfg: Dict, test_data: List[Dict], api_key: str) -> Dict:
    """Evaluasi satu model — handle thinking dan non-thinking."""
    if not HAS_LANGCHAIN:
        print("❌ langchain-openai not installed.")
        return {}

    model_id  = model_cfg["id"]
    is_think  = model_cfg["thinking"]
    prompt_fn = THINKING_PROMPT if is_think else STANDARD_PROMPT

    print(f"\n{'='*65}")
    print(f"📌 {model_name}  ({model_id})")
    print(f"   Thinking mode: {'✅ explicit <think>' if is_think else '❌ implicit (standard)'}")
    print(f"{'='*65}")

    llm = _make_llm(model_id, api_key)

    predictions   = []
    gt_labels, pred_labels = [], []
    gt_reasonings, pred_reasonings = [], []
    thinking_lengths = []
    inference_times  = []

    for i, sample in enumerate(test_data):
        try:
            gt_out      = json.loads(sample['output'])
            gt_label    = gt_out['label']
            gt_reasoning = str(gt_out.get('reasoning', ''))
        except Exception:
            continue

        title   = sample.get('title', sample['input'][:80])
        content = sample['input'][:600]
        prompt  = prompt_fn.format(title=title, content=content)

        t0 = time.time()
        try:
            raw = _call_with_retry(llm, prompt)
        except Exception as e:
            print(f"   ❌ Sample {i}: {e}")
            continue
        elapsed = time.time() - t0
        inference_times.append(elapsed)

        # Ekstrak thinking block (jika ada)
        thinking_process, rest = _extract_thinking(raw)
        thinking_lengths.append(len(thinking_process))

        parsed     = _parse_json(rest if rest else raw)
        pred_label = _normalize_label(parsed.get('label', ''))
        pred_reasoning = str(parsed.get('reasoning', ''))
        confidence     = float(parsed.get('confidence', 0.0))

        gt_labels.append(gt_label)
        pred_labels.append(pred_label)
        gt_reasonings.append(gt_reasoning)
        pred_reasonings.append(pred_reasoning)

        is_correct = (pred_label == gt_label)
        predictions.append({
            'model_name':        model_name,
            'input':             sample['input'][:300],
            'ground_truth':      gt_label,
            'prediction':        pred_label,
            'correct':           is_correct,
            'confidence':        confidence,
            'gt_reasoning':      gt_reasoning,
            'pred_reasoning':    pred_reasoning,
            'thinking_process':  thinking_process[:800] if thinking_process else '',
            'thinking_length':   len(thinking_process),
            'inference_time':    elapsed,
            'lang':              sample.get('lang', 'id'),
        })

        if (i + 1) % 10 == 0:
            acc = sum(1 for p in predictions if p['correct']) / len(predictions)
            avg_think = np.mean(thinking_lengths) if thinking_lengths else 0
            print(f"   [{i+1}/{len(test_data)}] Acc: {acc*100:.1f}% | "
                  f"Avg think len: {avg_think:.0f} chars | Avg time: {np.mean(inference_times):.2f}s")

    # ── Classification metrics ──
    result: Dict = {
        'model_name':         model_name,
        'model_id':           model_id,
        'is_thinking_model':  is_think,
        'family':             model_cfg['family'],
        'params':             model_cfg['params'],
        'num_samples':        len(predictions),
        'predictions':        predictions,
        'ground_truth_labels':gt_labels,
        'predicted_labels':   pred_labels,
        'reasoning_texts': {
            'ground_truth': gt_reasonings,
            'predicted':    pred_reasonings,
        },
        'thinking_stats': {
            'avg_length_chars': float(np.mean(thinking_lengths)) if thinking_lengths else 0,
            'max_length_chars': int(max(thinking_lengths)) if thinking_lengths else 0,
            'pct_with_thinking': float(np.mean([1 if t > 0 else 0 for t in thinking_lengths])) * 100,
        },
        'inference_speed': {
            'avg_time_seconds': float(np.mean(inference_times)) if inference_times else 0,
        },
    }

    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
        acc  = accuracy_score(gt_labels, pred_labels)
        prec, rec, f1, _ = precision_recall_fscore_support(
            gt_labels, pred_labels, average='weighted', zero_division=0
        )
        kappa = cohen_kappa_score(gt_labels, pred_labels)
        result['classification_metrics'] = {
            'accuracy': float(acc), 'precision': float(prec),
            'recall': float(rec),   'f1_score': float(f1),
            'cohens_kappa': float(kappa),
        }
        print(f"\n   📊 Acc={acc:.3f} | F1={f1:.3f} | κ={kappa:.3f}")
        if is_think:
            print(f"   🧠 Avg thinking length: {result['thinking_stats']['avg_length_chars']:.0f} chars "
                  f"({result['thinking_stats']['pct_with_thinking']:.0f}% samples had thinking)")
    except Exception as e:
        print(f"   ⚠️  sklearn metrics failed: {e}")
        result['classification_metrics'] = {}

    return result


# ─── NLP Metrics ─────────────────────────────────────────────────────────────

def compute_bertscore(result: Dict) -> Dict:
    if not HAS_BERTSCORE:
        return {}
    gt   = result['reasoning_texts']['ground_truth']
    pred = result['reasoning_texts']['predicted']
    valid = [(g, p) for g, p in zip(gt, pred) if g and p]
    if not valid:
        return {}
    gt_v, pred_v = zip(*valid)
    try:
        P, R, F1 = bert_score_fn(list(pred_v), list(gt_v), lang='id', verbose=False, device='cpu')
        return {'precision': float(P.mean()), 'recall': float(R.mean()), 'f1': float(F1.mean())}
    except Exception as e:
        print(f"   ⚠️  BERTScore failed: {e}")
        return {}


def compute_nlp_metrics(result: Dict) -> Dict:
    if not HAS_NLP:
        return {}
    gt   = result['reasoning_texts']['ground_truth']
    pred = result['reasoning_texts']['predicted']
    smoothie = SmoothingFunction().method1
    bleu_scores, meteor_scores = [], []
    for g, p in zip(gt, pred):
        if not g or not p:
            continue
        g_tok = nltk.word_tokenize(str(g).lower())
        p_tok = nltk.word_tokenize(str(p).lower())
        bleu_scores.append(sentence_bleu([g_tok], p_tok, smoothing_function=smoothie))
        try:
            meteor_scores.append(meteor_score([g_tok], p_tok))
        except Exception:
            pass
    return {
        'bleu':   float(np.mean(bleu_scores))   if bleu_scores   else 0.0,
        'meteor': float(np.mean(meteor_scores)) if meteor_scores else 0.0,
    }


# ─── LLM-as-a-Judge (aware of thinking) ──────────────────────────────────────

def run_judge(result: Dict, judge_model_id: str, api_key: str,
              max_samples: int = 30) -> Dict:
    """
    Judge yang tahu apakah model pakai thinking atau tidak.
    Menilai kualitas proses berpikir, bukan hanya akurasi label.
    """
    if not HAS_LANGCHAIN or not api_key:
        return {}

    judge_llm  = _make_llm(judge_model_id, api_key, temperature=0)
    model_name = result['model_name']
    is_think   = result['is_thinking_model']
    thinking_type = "explicit chain-of-thought (<think>)" if is_think else "implicit reasoning"

    preds = result['predictions']
    incorrect = [p for p in preds if not p['correct']]
    correct   = [p for p in preds if p['correct']]
    n_inc = min(int(max_samples * 0.8), len(incorrect))
    samples   = (incorrect[:n_inc] + correct)[:max_samples]

    scores, details, thinking_quality_dist = [], [], defaultdict(int)

    for i, pred in enumerate(samples):
        print(f"   Judging {i+1}/{len(samples)}...", end='\r')
        prompt = JUDGE_PROMPT.format(
            input_text       = pred['input'][:1200],
            ground_truth     = pred['ground_truth'],
            model_name       = model_name,
            thinking_type    = thinking_type,
            prediction       = pred['prediction'],
            thinking_process = (pred.get('thinking_process', '') or 'N/A')[:500],
            reasoning        = (pred.get('pred_reasoning', '') or 'N/A')[:300],
        )
        try:
            raw = _call_with_retry(judge_llm, prompt)
            parsed = _parse_json(raw)
            if not parsed and 'score' not in parsed:
                # Fallback: try extracting from raw
                m = re.search(r'"score"\s*:\s*(\d)', raw)
                if m:
                    parsed = {'score': int(m.group(1))}
            score = int(parsed.get('score', 0))
            tq    = parsed.get('thinking_quality', 'unknown')
            thinking_quality_dist[tq] += 1
            scores.append(score)
            details.append({
                'input_preview':   pred['input'][:150] + '...',
                'gt':              pred['ground_truth'],
                'pred':            pred['prediction'],
                'correct':         pred['correct'],
                'judge_score':     score,
                'justification':   parsed.get('justification', ''),
                'thinking_quality':tq,
                'had_thinking':    len(pred.get('thinking_process', '')) > 0,
                'thinking_length': pred.get('thinking_length', 0),
            })
        except Exception as e:
            print(f"\n   ⚠️  Judge error sample {i}: {e}")
            continue

    print()
    avg = float(np.mean(scores)) if scores else 0.0
    print(f"   🤖 Judge avg score: {avg:.2f}/5.0 ({len(scores)} samples)")
    print(f"   Thinking quality dist: {dict(thinking_quality_dist)}")

    return {
        'average_rating':          avg,
        'num_judged':              len(scores),
        'thinking_quality_dist':   dict(thinking_quality_dist),
        'judge_details':           details,
    }


# ─── Qualitative Examples Export ─────────────────────────────────────────────

def export_qualitative_examples(result: Dict, judge_result: Dict,
                                 output_path: str, n: int = 3):
    """
    Export contoh reasoning/thinking secara kualitatif:
    good (4-5), borderline (3), bad (1-2).
    Termasuk thinking_process untuk model yang punya <think>.
    """
    pred_lookup = {p['input'][:100]: p for p in result.get('predictions', [])}
    details = judge_result.get('judge_details', [])
    if not details:
        return

    good, borderline, bad = [], [], []
    for d in details:
        score = d.get('judge_score', 0)
        key   = d.get('input_preview', '')[:100]
        full  = pred_lookup.get(key, {})
        entry = {
            'judge_score':        score,
            'judge_justification':d.get('justification', ''),
            'thinking_quality':   d.get('thinking_quality', ''),
            'ground_truth_label': d.get('gt', ''),
            'predicted_label':    d.get('pred', ''),
            'correct':            d.get('correct', False),
            'had_thinking':       d.get('had_thinking', False),
            'thinking_length_chars': d.get('thinking_length', 0),
            'article_preview':    full.get('input', '')[:400],
            'thinking_process':   (full.get('thinking_process', '') or '')[:600],
            'model_reasoning':    full.get('pred_reasoning', '')[:400],
            'ground_truth_reasoning': full.get('gt_reasoning', '')[:400],
            'confidence':         full.get('confidence', None),
        }
        if score >= 4:
            good.append(entry)
        elif score == 3:
            borderline.append(entry)
        else:
            bad.append(entry)

    output = {
        '_description': (
            f"Qualitative reasoning examples for {result['model_name']} "
            f"({'explicit <think>' if result['is_thinking_model'] else 'implicit reasoning'}). "
            "Groups: good=4-5, borderline=3, bad=1-2."
        ),
        'model_info': {
            'name':       result['model_name'],
            'is_thinking':result['is_thinking_model'],
            'family':     result['family'],
        },
        'summary': {
            'total_judged': len(details),
            'good_count':   len([d for d in details if d.get('judge_score', 0) >= 4]),
            'borderline_count': len([d for d in details if d.get('judge_score', 0) == 3]),
            'bad_count':    len([d for d in details if d.get('judge_score', 0) <= 2]),
            'avg_score':    round(judge_result.get('average_rating', 0), 3),
            'thinking_quality_distribution': judge_result.get('thinking_quality_dist', {}),
        },
        'good_examples':       sorted(good,       key=lambda x: -x['judge_score'])[:n],
        'borderline_examples': sorted(borderline, key=lambda x: -x['judge_score'])[:n],
        'bad_examples':        sorted(bad,         key=lambda x:  x['judge_score'])[:n],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"   📋 Reasoning examples: {output_path}")


# ─── Analysis: Explicit vs Implicit ─────────────────────────────────────────

def analyze_thinking_vs_nothinking(all_results: List[Dict]) -> Dict:
    """
    Bandingkan model dengan explicit thinking vs implicit reasoning:
    - Rata-rata panjang thinking
    - Akurasi, F1, Cohen's κ
    - BERTScore, BLEU, METEOR (reasoning summary)
    - Judge score
    """
    thinking_group    = [r for r in all_results if r.get('is_thinking_model')]
    nothinking_group  = [r for r in all_results if not r.get('is_thinking_model')]

    def _stats(group: List[Dict]) -> Dict:
        if not group:
            return {}
        cm      = [r.get('classification_metrics', {}) for r in group]
        accs    = [c.get('accuracy', 0) for c in cm]
        f1s     = [c.get('f1_score',  0) for c in cm]
        kappas  = [c.get('cohens_kappa', 0) for c in cm]
        bs_f1s  = [r.get('bertscore', {}).get('f1', 0)     for r in group if r.get('bertscore')]
        bleus   = [r.get('nlp_metrics', {}).get('bleu', 0) for r in group if r.get('nlp_metrics')]
        meteors = [r.get('nlp_metrics', {}).get('meteor',0) for r in group if r.get('nlp_metrics')]
        judges  = [r.get('judge_results', {}).get('average_rating', 0)
                   for r in group if r.get('judge_results')]
        think_lens = [r.get('thinking_stats', {}).get('avg_length_chars', 0) for r in group]
        return {
            'models':                   [r['model_name'] for r in group],
            'avg_accuracy':             float(np.mean(accs))    if accs    else 0,
            'avg_f1':                   float(np.mean(f1s))     if f1s     else 0,
            'avg_kappa':                float(np.mean(kappas))  if kappas  else 0,
            'avg_bertscore_f1':         float(np.mean(bs_f1s))  if bs_f1s  else 0,
            'avg_bleu':                 float(np.mean(bleus))   if bleus   else 0,
            'avg_meteor':               float(np.mean(meteors)) if meteors else 0,
            'avg_judge_score':          float(np.mean(judges))  if judges  else 0,
            'avg_thinking_len_chars':   float(np.mean(think_lens)),
        }

    analysis = {
        'thinking_group':    _stats(thinking_group),
        'nothinking_group':  _stats(nothinking_group),
    }

    print(f"\n{'='*65}")
    print("🧠 EXPLICIT THINKING vs IMPLICIT REASONING ANALYSIS")
    print(f"{'='*65}")
    for label, stats in [("Explicit Thinking (Qwen3, DeepSeek-R1)", analysis['thinking_group']),
                          ("Implicit Reasoning (Gemma 3 12B)", analysis['nothinking_group'])]:
        if not stats:
            continue
        print(f"\n  [{label}]")
        print(f"  Models       : {stats['models']}")
        print(f"  Accuracy     : {stats['avg_accuracy']:.4f}")
        print(f"  F1-Score     : {stats['avg_f1']:.4f}")
        print(f"  Cohen's κ    : {stats['avg_kappa']:.4f}")
        print(f"  BERTScore F1 : {stats['avg_bertscore_f1']:.4f}")
        print(f"  BLEU         : {stats['avg_bleu']:.4f}")
        print(f"  METEOR       : {stats['avg_meteor']:.4f}")
        print(f"  Judge Score  : {stats['avg_judge_score']:.2f}/5.0")
        print(f"  Avg Think Len: {stats['avg_thinking_len_chars']:.0f} chars")

    return analysis


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_results(all_results: List[Dict], analysis: Dict, output_dir: Path):
    if not HAS_PLOT:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names  = [r['model_name'] for r in all_results]
    is_thinking  = [r['is_thinking_model'] for r in all_results]
    # Color: blue for thinking, red for non-thinking
    colors = ['#4A90D9' if t else '#E74C3C' for t in is_thinking]
    labels_patch = [
        plt.matplotlib.patches.Patch(color='#4A90D9', label='Explicit Thinking'),
        plt.matplotlib.patches.Patch(color='#E74C3C', label='Implicit Reasoning'),
    ]

    # ── Plot 1: Classification metrics 3-bar ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('Classification Performance: Thinking vs Non-Thinking Models',
                 fontsize=13, fontweight='bold', y=1.02)
    for ax, (key, lbl) in zip(axes, [
        ('accuracy', 'Accuracy'), ('f1_score', 'F1-Score (weighted)'), ('cohens_kappa', "Cohen's κ")
    ]):
        vals = [r.get('classification_metrics', {}).get(key, 0) for r in all_results]
        bars = ax.bar(model_names, vals, color=colors, edgecolor='white', linewidth=1.2)
        ax.set_title(lbl, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='x', rotation=20)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(handles=labels_patch, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / 'classification_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 2: Thinking length (chars) per model ──
    think_lens = [r.get('thinking_stats', {}).get('avg_length_chars', 0) for r in all_results]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(model_names, think_lens, color=colors, edgecolor='white', linewidth=1.2)
    ax.set_title('Average Thinking Process Length per Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg chars in <think> block')
    ax.tick_params(axis='x', rotation=15)
    for bar, v in zip(bars, think_lens):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{v:.0f}' if v > 0 else 'N/A', ha='center', va='bottom', fontsize=9)
    ax.legend(handles=labels_patch)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'thinking_length.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 3: Radar Chart — Thinking vs Non-Thinking ──
    t_stats  = analysis.get('thinking_group', {})
    nt_stats = analysis.get('nothinking_group', {})
    if t_stats and nt_stats:
        cats   = ['Accuracy', 'F1', "Cohen's κ", 'BERTScore', 'BLEU', 'METEOR', 'Judge (÷5)']
        t_vals = [t_stats['avg_accuracy'], t_stats['avg_f1'], max(0, t_stats['avg_kappa']),
                  t_stats['avg_bertscore_f1'], t_stats['avg_bleu'], t_stats['avg_meteor'],
                  t_stats['avg_judge_score'] / 5]
        nt_vals= [nt_stats['avg_accuracy'], nt_stats['avg_f1'], max(0, nt_stats['avg_kappa']),
                  nt_stats['avg_bertscore_f1'], nt_stats['avg_bleu'], nt_stats['avg_meteor'],
                  nt_stats['avg_judge_score'] / 5]
        N      = len(cats)
        angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
        t_vals  += t_vals[:1]
        nt_vals += nt_vals[:1]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cats, size=10)
        ax.set_ylim(0, 1)
        ax.plot(angles, t_vals,  'o-', lw=2, color='#4A90D9',
                label=f"Explicit Thinking ({', '.join(t_stats.get('models', []))})")
        ax.fill(angles, t_vals,  alpha=0.2, color='#4A90D9')
        ax.plot(angles, nt_vals, 'o-', lw=2, color='#E74C3C',
                label=f"Implicit Reasoning ({', '.join(nt_stats.get('models', []))})")
        ax.fill(angles, nt_vals, alpha=0.2, color='#E74C3C')
        ax.set_title('Explicit Thinking vs Implicit Reasoning\nComprehensive Comparison',
                     size=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.1))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'radar_thinking_vs_nothinking.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ radar_thinking_vs_nothinking.png")

    print(f"✅ Plots saved to {output_dir}/")


# ─── Export ───────────────────────────────────────────────────────────────────

def export_csv_table(all_results: List[Dict], output_path: str):
    if not HAS_PANDAS:
        return
    rows = []
    for r in all_results:
        cm  = r.get('classification_metrics', {})
        bs  = r.get('bertscore', {})
        nlp = r.get('nlp_metrics', {})
        jd  = r.get('judge_results', {})
        th  = r.get('thinking_stats', {})
        spd = r.get('inference_speed', {})
        rows.append({
            'Model':              r['model_name'],
            'Family':             r['family'],
            'Params':             r['params'],
            'Explicit Thinking':  r['is_thinking_model'],
            'Accuracy':           round(cm.get('accuracy', 0), 4),
            'F1-Score':           round(cm.get('f1_score', 0), 4),
            "Cohen's κ":          round(cm.get('cohens_kappa', 0), 4),
            'BERTScore F1':       round(bs.get('f1', 0), 4),
            'BLEU':               round(nlp.get('bleu', 0), 4),
            'METEOR':             round(nlp.get('meteor', 0), 4),
            'Judge Score (÷5)':   round(jd.get('average_rating', 0), 2),
            'Avg Think Len (chr)':round(th.get('avg_length_chars', 0)),
            'Avg Infer (s)':      round(spd.get('avg_time_seconds', 0), 3),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n{'='*65}")
    print("MODEL COMPARISON TABLE")
    print(f"{'='*65}")
    print(df.to_string(index=False))
    print(f"\n✅ CSV: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Reasoning Model Evaluation — Explicit Thinking vs Implicit Reasoning'
    )
    parser.add_argument('--dataset', type=str,
                        default='../data/llm_dataset_finetuning_optimized.json')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Jumlah sampel test per model')
    parser.add_argument('--output', type=str, default='reasoning_model_results')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenRouter API key')
    parser.add_argument('--judge-model', type=str,
                        default='google/gemma-3-12b-it',
                        help='Model untuk LLM-as-a-Judge')
    parser.add_argument('--max-judge-samples', type=int, default=30)
    parser.add_argument('--skip-judge', action='store_true',
                        help='Skip LLM-as-a-Judge (lebih cepat)')
    parser.add_argument('--models', type=str, default='all',
                        help='Model yang dievaluasi: all | Qwen,DeepSeek,Gemma')
    args = parser.parse_args()

    api_key = (args.api_key
               or os.environ.get('OPENROUTER_API_KEY')
               or os.environ.get('openrouter'))
    if not api_key:
        print("❌ API key required: --api-key sk-or-v1-xxx")
        print("   atau set env OPENROUTER_API_KEY")
        sys.exit(1)

    # Pilih model
    if args.models == 'all':
        selected = MODELS
    else:
        kws = [m.strip().lower() for m in args.models.split(',')]
        selected = {n: cfg for n, cfg in MODELS.items()
                    if any(kw in n.lower() for kw in kws)}
        if not selected:
            print(f"❌ Tidak ada model cocok: {args.models}")
            sys.exit(1)

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("🔬 REASONING MODEL EVALUATION")
    print("=" * 65)
    print(f"Dataset : {args.dataset}")
    print(f"Samples : {args.num_samples}")
    print(f"Output  : {output_dir}")
    print(f"Judge   : {args.judge_model} (skip={args.skip_judge})")
    print("Models:")
    for name, cfg in selected.items():
        tag = "🧠 [THINKING]" if cfg['thinking'] else "💬 [STANDARD]"
        print(f"  {tag} {name} ({cfg['params']})  → {cfg['id']}")
    print("=" * 65 + "\n")

    # Load data
    print(f"📂 Loading dataset...")
    test_data = load_test_samples(args.dataset, args.num_samples)
    print(f"✅ {len(test_data)} samples loaded\n")

    all_results = []
    for model_name, model_cfg in selected.items():
        res = evaluate_model(model_name, model_cfg, test_data, api_key)
        if not res:
            continue

        # NLP metrics
        print(f"\n   📝 Computing NLP metrics for {model_name}...")
        res['bertscore']   = compute_bertscore(res)
        res['nlp_metrics'] = compute_nlp_metrics(res)
        if res['bertscore']:
            print(f"   BERTScore F1: {res['bertscore']['f1']:.4f}")
        if res['nlp_metrics']:
            print(f"   BLEU: {res['nlp_metrics']['bleu']:.4f} | METEOR: {res['nlp_metrics']['meteor']:.4f}")

        # Judge
        if not args.skip_judge:
            print(f"\n   🤖 Running judge for {model_name}...")
            judge_out = run_judge(res, args.judge_model, api_key, args.max_judge_samples)
            res['judge_results'] = judge_out

            # Qualitative examples
            ex_path = output_dir / f'reasoning_examples_{model_name.replace(" ","_")}_{timestamp}.json'
            export_qualitative_examples(res, judge_out, str(ex_path))
        else:
            res['judge_results'] = {}

        all_results.append(res)

    if not all_results:
        print("❌ Tidak ada model yang berhasil dievaluasi.")
        return

    # Analysis
    analysis = analyze_thinking_vs_nothinking(all_results)

    # Plots
    print(f"\n📊 Creating visualizations...")
    plot_results(all_results, analysis, output_dir)

    # Export summary JSON
    summary_path = output_dir / f'summary_{timestamp}.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'analysis': analysis,
            'models': [{k: v for k, v in r.items()
                        if k not in ('predictions', 'ground_truth_labels',
                                     'predicted_labels', 'reasoning_texts')}
                       for r in all_results]
        }, f, ensure_ascii=False, indent=2, default=str)

    # CSV
    csv_path = output_dir / f'comparison_{timestamp}.csv'
    export_csv_table(all_results, str(csv_path))

    print(f"\n{'='*65}")
    print("🎉 EVALUATION COMPLETE!")
    print(f"{'='*65}")
    print(f"  📝 {summary_path}")
    print(f"  📊 {output_dir}/classification_metrics.png")
    print(f"  📊 {output_dir}/thinking_length.png")
    print(f"  📊 {output_dir}/radar_thinking_vs_nothinking.png")
    print(f"  📋 reasoning_examples_*_{timestamp}.json  (per model)")
    print(f"  📋 {csv_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
