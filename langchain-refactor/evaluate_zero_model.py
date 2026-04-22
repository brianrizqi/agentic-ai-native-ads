"""
Scenario 0: Zero-Shot Model Evaluation (No Fine-tuning, No RAG, No Domain Prompt)
==================================================================================
Mirror of evaluate_model.py — same load_test_set(), same metrics pipeline,
same LLM-as-a-Judge, same output format — but loads models via Unsloth
(off-the-shelf, no LoRA, no fine-tuned weights) and uses a minimal binary prompt.

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

import json
import argparse
import os
import re
import time
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from datetime import datetime

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

try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    print("Warning: unsloth not installed. pip install unsloth")
    print("Falling back to standard HuggingFace transformers.")

if not HAS_UNSLOTH:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        HAS_TRANSFORMERS = True
    except ImportError:
        HAS_TRANSFORMERS = False
        print("Error: Neither unsloth nor transformers is installed.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot prompts — minimal binary question, no domain framing
# Mirrors Hu et al. (AAAI 2024) zero-shot setup for fair comparison
# ─────────────────────────────────────────────────────────────────────────────

ZERO_SHOT_PROMPT_ID = """Baca artikel berita berikut dengan seksama dan tentukan apakah artikel tersebut merupakan native ads atau berita murni.

Native ads adalah konten berbayar yang dirancang menyerupai berita editorial namun memiliki tujuan komersial.
Berita murni adalah laporan editorial yang tidak memiliki agenda komersial.

Artikel:
{article}

Jawab dengan tepat salah satu label berikut: native ads atau berita murni.
Label:"""

ZERO_SHOT_PROMPT_EN = """Read the following news article carefully and determine whether it is native advertising or pure news.

Native advertising is sponsored content designed to resemble editorial reporting while serving a commercial purpose.
Pure news is legitimate editorial reporting with no commercial agenda.

Article:
{article}

Answer with exactly one of the following labels: Native Ads or Pure News.
Label:"""


def build_zero_shot_prompt(article_text: str, lang: str) -> str:
    """Build minimal binary prompt — no four-dimension framing, no retrieved examples."""
    article_text = article_text.strip()
    if lang == "en":
        return ZERO_SHOT_PROMPT_EN.format(article=article_text)
    return ZERO_SHOT_PROMPT_ID.format(article=article_text)


# ─────────────────────────────────────────────────────────────────────────────
# Label parsing — handles mixed-language model outputs
# ─────────────────────────────────────────────────────────────────────────────

NATIVE_ADS_PATTERNS = [
    r"\bnative[\s_\-]*ads?\b",
    r"\bnative[\s_\-]*advertising\b",
    r"\biklan[\s_\-]*native\b",
    r"\bsponsor(ed)?\b",
    r"\bpromosi(onal)?\b",
]

PURE_NEWS_PATTERNS = [
    r"\bpure[\s_\-]*news\b",
    r"\bberita[\s_\-]*murni\b",
    r"\blegitimate[\s_]*editorial\b",
    r"\bbukan[\s_]*iklan\b",
]


def parse_zero_shot_label(raw_output: str, lang: str) -> str:
    """
    Extract predicted label from raw model output.
    Returns 'native ads', 'berita murni', or 'Unknown'.
    Uses normalized strings so downstream metrics match evaluate_model.py label space.
    """
    text = raw_output.lower().strip()

    # Native ads check first (more distinctive patterns)
    for p in NATIVE_ADS_PATTERNS:
        if re.search(p, text):
            return "native ads" if lang == "id" else "Native Ads"

    for p in PURE_NEWS_PATTERNS:
        if re.search(p, text):
            return "berita murni" if lang == "id" else "Pure News"

    # Fallback: first meaningful token
    first = text.split()[0] if text.split() else ""
    if first in {"native", "iklan", "sponsored", "promosi"}:
        return "native ads" if lang == "id" else "Native Ads"
    if first in {"pure", "berita", "murni", "legitimate", "editorial", "news"}:
        return "berita murni" if lang == "id" else "Pure News"

    return "Unknown"


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
        import random
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

        import random
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
    import random
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
    Falls back to standard HuggingFace if Unsloth is not available.
    """
    import torch

    if HAS_UNSLOTH:
        print(f"🦥 Loading via Unsloth: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,          # Auto-detect (bfloat16 on Ampere+, float16 otherwise)
            load_in_4bit=True,   # 4-bit quant — same memory footprint as fine-tuned eval
        )
        FastLanguageModel.for_inference(model)
        print(f"✅ Unsloth model loaded (4-bit, inference mode)")

    else:
        print(f"🤗 Unsloth not found. Loading via HuggingFace transformers: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        print(f"✅ HuggingFace model loaded (float16, device_map=auto)")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot inference
# ─────────────────────────────────────────────────────────────────────────────

import torch

@torch.no_grad()
def zero_shot_classify(
    model,
    tokenizer,
    article_text: str,
    lang: str,
    max_new_tokens: int = 64,
    max_article_chars: int = 3000,
) -> Dict:
    """
    Run zero-shot classification on a single article.
    Returns dict with label, confidence (placeholder 1.0), reasoning (raw output).
    """
    # Truncate very long articles to fit context window
    if len(article_text) > max_article_chars:
        article_text = article_text[:max_article_chars] + " [...]"

    prompt = build_zero_shot_prompt(article_text, lang)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
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
    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    label = parse_zero_shot_label(raw_text, lang)

    return {
        "label":     label,
        "confidence": 1.0,   # Zero-shot has no calibrated confidence
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
            pred = zero_shot_classify(
                model, tokenizer,
                article_text=sample['input'],
                lang=lang,
                max_new_tokens=kwargs.get('max_new_tokens', 64),
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
                "X-Title": "Agentic Native Ads Evaluation — Scenario 0 Zero-Shot",
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
    parser.add_argument('--max-new-tokens', type=int, default=64,
                        help='Max tokens to generate per sample (default: 64)')
    parser.add_argument('--max-seq-length', type=int, default=2048,
                        help='Max sequence length for Unsloth (default: 2048)')
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
