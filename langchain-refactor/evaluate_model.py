"""
Comprehensive Model Evaluation Script
- Confusion Matrix with visualization
- BERTScore for reasoning quality
- LLM-as-a-Judge for qualitative assessment
- Standard classification metrics
"""

import json
import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

sys.path.append(str(Path(__file__).parent))

from agents.classification_agent import ClassificationAgent

# For visualization and metrics
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn/matplotlib not installed. Install with: pip install scikit-learn matplotlib seaborn")

# For BERTScore
try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("Warning: bert-score not installed. Install with: pip install bert-score")

# For LLM-as-a-Judge
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# For NLP Metrics
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"Note: Could not download NLTK data (likely no internet): {e}")
    HAS_NLP_METRICS = True
except ImportError:
    HAS_NLP_METRICS = False
    print("Warning: nltk not installed. Install with: pip install nltk")


def load_test_set(dataset_path: str, num_samples: int = 100, lang_filter: str = None) -> List[Dict]:
    """Load test samples from dataset. Shuffles with seed to ensure diversity."""
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter by language if specified
    if lang_filter:
        data = [item for item in data if item.get('lang') == lang_filter.lower()]
        print(f"🌍 Filtered by language: {lang_filter} ({len(data)} samples found)")
        # Shuffle and take samples
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
        
        print(f"🌍 No language filter. Sampling proportionally from {len(lang_groups)} languages...")
        
        import random
        random.seed(42)
        
        for lang, items in lang_groups.items():
            # Calculate proportion
            proportion = len(items) / total_available
            lang_samples = int(num_samples * proportion)
            
            random.shuffle(items)
            test_data.extend(items[:lang_samples])
            print(f"   - {lang}: {lang_samples} samples")
            
        # If we're slightly under due to rounding, fill from the largest group
        if len(test_data) < num_samples:
            diff = num_samples - len(test_data)
            largest_lang = max(lang_groups, key=lambda k: len(lang_groups[k]))
            # Get samples not already in test_data
            current_ids = {item['id'] for item in test_data}
            remaining = [item for item in lang_groups[largest_lang] if item['id'] not in current_ids]
            test_data.extend(remaining[:diff])

    # Final shuffle of the mixed set
    import random
    random.seed(42)
    random.shuffle(test_data)
    
    # Print distribution
    from collections import Counter
    dist = Counter()
    for item in test_data:
        try:
            output = json.loads(item['output'])
            dist[output['label']] += 1
        except:
            pass
    print(f"📊 Test set distribution: {dict(dist)}")
    
    return test_data


def knn_vote_label(neighbors: List[Dict]) -> Tuple[Optional[str], float, float]:
    """Weighted KNN vote over L2-distance neighbors (lower distance = closer).

    Returns (winning_label, margin in [0,1], min_distance). Used by --rag-vote to
    correct the model post-hoc WITHOUT touching its (in-distribution) prompt.
    """
    if not neighbors:
        return None, 0.0, 99.0
    vote = {'native ads': 0.0, 'berita murni': 0.0}
    min_dist = 99.0
    for n in neighbors:
        l2 = float(n.get('similarity_score', 2.0))
        min_dist = min(min_dist, l2)
        w = max(0.0, 1.0 - l2 / 2.0)  # L2 in [0,2] -> weight in [0,1]
        lab = 'native ads' if 'native' in str(n.get('label', '')).lower() else 'berita murni'
        vote[lab] += w
    total = vote['native ads'] + vote['berita murni']
    if total <= 0:
        return None, 0.0, min_dist
    if vote['native ads'] >= vote['berita murni']:
        return 'native ads', (vote['native ads'] - vote['berita murni']) / total, min_dist
    return 'berita murni', (vote['berita murni'] - vote['native ads']) / total, min_dist


def evaluate_model(model_path: str, test_data: List[Dict], lora_path: Optional[str] = None, **kwargs) -> Dict:
    """Evaluate model on test set with comprehensive metrics."""
    
    print(f"Initializing model from: {model_path}")
    if lora_path:
        print(f"Using LoRA adapters from: {lora_path}")
    
    agent = ClassificationAgent(
        provider='local', 
        model_name=model_path,
        lora_path=lora_path,
        gpu_id=kwargs.get('gpu_id', 0),
        temperature=0.0, # Phase 48: Force 0.0 for deterministic evaluation
        use_rag=kwargs.get('use_rag', False),
        is_mcq=kwargs.get('is_mcq', False), # Phase 47: Stop forcing MCQ via the caller
        model_tier=kwargs.get('tier', 'base'), # Phase 40: Pass tier from CLI
        rag_top_k=kwargs.get('top_k', 3) # Stage 75: Support Top-K from CLI
    )
    
    # Initialize RAG if requested
    retriever = None
    if kwargs.get('use_rag'):
        try:
            from agents.retrieval_agent import RetrievalAgent
            from vector_stores.native_ads_vectorstore import NativeAdsVectorStore
            
            print(f"🔍 Initializing RAG with vectorstore: {kwargs.get('vectorstore_dir')}")
            vstore = NativeAdsVectorStore(
                embedding_model=kwargs.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2"),
                persist_directory=kwargs.get('vectorstore_dir')
            )
            retriever = RetrievalAgent(vectorstore=vstore)
        except ImportError as e:
            print(f"⚠️  RAG initialization failed: {e}")
            print("Please install missing dependencies: pip install langchain-community faiss-cpu")
            print("Continuing evaluation without RAG...")

    results = {

        'correct': 0,
        'total': 0,
        'by_class': defaultdict(lambda: {'correct': 0, 'total': 0, 'predictions': []}),
        'confusion_matrix': defaultdict(int),
        'predictions': [],
        'ground_truth_labels': [],
        'predicted_labels': [],
        'reasoning_texts': {'ground_truth': [], 'predicted': []},
        'perplexities': []
    }
    
    print(f"\nEvaluating on {len(test_data)} samples...")
    
    for i, sample in enumerate(test_data):
        # Get ground truth
        try:
            gt_data = json.loads(sample['output'])
            gt_label = gt_data['label']
            gt_reasoning = gt_data.get('reasoning', '')
        except:
            continue
        
        # Get prediction
        try:
            title = sample.get('title', sample['input'][:100])
            
            # Get RAG context if enabled
            context = ""
            examples = None
            if retriever:
                # Phase 220: Respect --top-k EXACTLY. For train-time RAG the eval top_k MUST
                # equal the --rag-top-k used during fine-tuning (default 4), otherwise the
                # number of embedded examples differs from training and breaks alignment.
                eff_top_k = kwargs.get('top_k', 4)
                max_reasoning = None
                minimalist = False
                
                # Phase 20: Use examples for multi-turn history-based RAG for local models
                # Phase 200: Exception — Gemma large (12B+) uses llm.invoke() pipeline,
                # which reads from `context` string, NOT from `examples` list.
                # The March 25 winning run (98.5%) used get_context_for_classification().
                is_gemma_large = (
                    agent.provider == "local" and agent.tokenizer and
                    "gemma" in getattr(agent, 'model_name', '').lower() and
                    getattr(agent, 'model_tier', '') == 'standard'
                )
                if is_gemma_large:
                    # Restore the exact March 25 RAG path: context string with Label+Alasan+Isi
                    context = retriever.get_context_for_classification(
                        sample['input'],
                        top_k=eff_top_k,
                        max_reasoning_length=max_reasoning,
                        minimalist=minimalist
                    )
                elif agent.provider == "local" and agent.tokenizer:
                    # Balanced retrieval: equal native ads + berita murni examples.
                    # Plain top-K retrieval biases toward whichever class dominates the
                    # embedding neighborhood, causing systematic over-prediction of that class.
                    examples = retriever.get_balanced_examples_for_history(
                        sample['input'],
                        top_k=eff_top_k
                    )
                else:
                    context = retriever.get_context_for_classification(
                        sample['input'], 
                        top_k=eff_top_k,
                        max_reasoning_length=max_reasoning,
                        minimalist=minimalist
                    )
                
            pred = agent.classify(sample['input'], title=title, context=context, examples=examples)

            pred_label = pred.get('label', 'unknown')
            pred_reasoning = pred.get('reasoning', '')
            
            # Ensure reasoning is a string (could be a list from model output)
            if isinstance(pred_reasoning, list):
                pred_reasoning = " ".join([str(i) for i in pred_reasoning])
            else:
                pred_reasoning = str(pred_reasoning)
                
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue
        
        # Ensure gt_reasoning is also a string
        if isinstance(gt_reasoning, list):
            gt_reasoning = " ".join([str(i) for i in gt_reasoning])
        else:
            gt_reasoning = str(gt_reasoning)
            
        # Update metrics
        results['total'] += 1
        results['by_class'][gt_label]['total'] += 1
        results['ground_truth_labels'].append(gt_label)
        results['predicted_labels'].append(pred_label)
        
        # Store reasoning for NLP metrics
        results['reasoning_texts']['ground_truth'].append(gt_reasoning)
        results['reasoning_texts']['predicted'].append(pred_reasoning)
        
        # Calculate perplexity if local model
        if hasattr(agent, 'compute_perplexity'):
            metadata = pred.get('metadata', {})
            # Phase 69: Prioritize real-time logit-based PPL from the agent
            if 'ppl' in metadata:
                ppl = metadata['ppl']
            else:
                ppl_text = metadata.get('raw_response', pred_reasoning)
                ppl_prompt = metadata.get('raw_prompt', "")
                ppl = agent.compute_perplexity(ppl_text, prompt=ppl_prompt)
            
            if ppl > 0:
                results['perplexities'].append(ppl)
        
        if pred_label == gt_label:
            results['correct'] += 1
            results['by_class'][gt_label]['correct'] += 1
        
        results['confusion_matrix'][f"{gt_label} -> {pred_label}"] += 1
        results['by_class'][gt_label]['predictions'].append({
            'input': sample['input'][:200],
            'predicted': pred_label,
            'confidence': pred.get('confidence', 0),
            'correct': pred_label == gt_label
        })
        
        results['predictions'].append({
            'input': sample['input'][:200],
            'ground_truth': gt_label,
            'prediction': pred_label,
            'confidence': pred.get('confidence', 0),
            'gt_reasoning': gt_reasoning,
            'pred_reasoning': pred_reasoning,
            'correct': pred_label == gt_label
        })
        
        # Phase 40: Real-time progress (every sample)
        print(f"[{i + 1}/{len(test_data)}] {gt_label} -> {pred_label}" + (f" (PPL: {ppl:.2f})" if 'ppl' in locals() else ""))
    
    return results


def compute_bertscore(results: Dict) -> Dict:
    """Compute BERTScore for reasoning quality."""
    
    if not HAS_BERTSCORE:
        return {}
    
    print("\n📊 Computing BERTScore for reasoning quality (CPU mode)...")
    
    gt_reasoning = results['reasoning_texts']['ground_truth']
    pred_reasoning = results['reasoning_texts']['predicted']
    
    # Filter out empty strings
    valid_pairs = [(gt, pred) for gt, pred in zip(gt_reasoning, pred_reasoning) 
                   if gt and pred]
    
    if not valid_pairs:
        return {}
    
    gt_texts, pred_texts = zip(*valid_pairs)
    gt_texts = [str(t) for t in gt_texts]
    pred_texts = [str(t) for t in pred_texts]
    
    try:
        P, R, F1 = bert_score(pred_texts, gt_texts,
                               model_type='bert-base-multilingual-cased',
                               verbose=False, device='cpu')
        return {
            'precision': float(P.mean()),
            'recall': float(R.mean()),
            'f1': float(F1.mean())
        }
    except Exception as e:
        print(f"⚠️  BERTScore failed: {e}")
        return {}


def compute_nlp_metrics(results: Dict) -> Dict:
    """Compute BLEU, ROUGE, and METEOR for reasoning quality."""
    
    if not HAS_NLP_METRICS:
        return {}
    
    print("\n📝 Computing BLEU, ROUGE, and METEOR metrics...")
    
    gt_reasoning = results['reasoning_texts']['ground_truth']
    pred_reasoning = results['reasoning_texts']['predicted']
    
    bleu_scores = []
    meteor_scores = []
    
    smoothie = SmoothingFunction().method1
    
    processed_count = 0
    for gt, pred in zip(gt_reasoning, pred_reasoning):
        if not gt or not pred:
            continue
            
        processed_count += 1
        # Tokenize for NLTK
        gt_lower = str(gt).lower()
        pred_lower = str(pred).lower()
        
        gt_tokens = nltk.word_tokenize(gt_lower)
        pred_tokens = nltk.word_tokenize(pred_lower)
        
        # BLEU
        bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu)
        
        # METEOR
        try:
            meteor = meteor_score([gt_tokens], pred_tokens)
            meteor_scores.append(meteor)
        except Exception:
            pass
            
    print(f"   (Processed {processed_count} samples for NLP metrics)")
            
    return {
        'bleu': np.mean(bleu_scores) if bleu_scores else 0,
        'meteor': np.mean(meteor_scores) if meteor_scores else 0
    }


def llm_as_judge(results: Dict, api_key: str = None, provider: str = "openai", model: str = "gpt-4o-mini") -> Dict:
    """Use an LLM (OpenAI or OpenRouter) to judge prediction quality, prioritizing errors."""
    
    if not HAS_OPENAI:
        print("⚠️  Skipping LLM-as-a-Judge: langchain-openai not installed.")
        return {}

    if not api_key:
        print(f"⚠️  Skipping LLM-as-a-Judge: API Key missing. Please provide via --api-key.")
        return {}
    
    final_api_key = api_key
    
    print(f"\n🤖 Running LLM-as-a-Judge evaluation (Provider: {provider}, Model: {model})...")
    
    if provider == "openrouter":
        llm = ChatOpenAI(
            model=model,
            openai_api_key=final_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            default_headers={
                "HTTP-Referer": "https://github.com/brianrizqi/agentic-ai-native-ads",
                "X-Title": "Agentic Native Ads Evaluation"
            }
        )
    else:
        llm = ChatOpenAI(
            model=model,
            openai_api_key=final_api_key,
            temperature=0
        )
    
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
        input_variables=["input_text", "ground_truth", "prediction", "reasoning"]
    )
    
    all_predictions = results['predictions']
    # Prioritize judging incorrect classifications
    incorrect = [p for p in all_predictions if not p.get('correct')]
    correct = [p for p in all_predictions if p.get('correct')]
    
    sample_size = min(100, len(all_predictions))
    # Mix: up to 80% incorrect, 20% correct
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
                    reasoning=pred.get('pred_reasoning', '')
                )
            )
            
            judge_result = json.loads(response.content)
            ratings.append(judge_result['rating'])
            details.append({
                'input_preview': pred['input'][:100] + "...",
                'gt': pred['ground_truth'],
                'pred': pred['prediction'],
                'judge_rating': judge_result['rating'],
                'judge_justification': judge_result['justification']
            })
        except Exception as e:
            print(f"\n   Judge error on sample {i}: {e}")
            continue
    
    print(f"\n✅ Finished judging {len(ratings)} samples.")
    
    return {
        'average_rating': np.mean(ratings) if ratings else 0,
        'num_samples_judged': len(ratings),
        'judge_details': details
    }


def export_reasoning_examples(
    results: Dict,
    judge_results: Dict,
    output_path: str,
    n_per_category: int = 3
) -> None:
    """
    Export contoh-contoh reasoning model secara kualitatif ke JSON.
    Dikelompokkan menjadi:
      - good       : judge score 4-5 (reasoning mendalam & label benar)
      - borderline : judge score 3   (label benar, reasoning lemah)
      - bad        : judge score 1-2 (reasoning salah / label salah)
    
    Format output cocok untuk screenshot/dokumentasi di research paper.
    """

    # Bangun lookup: input_preview -> detail prediksi lengkap
    pred_lookup = {}
    for p in results.get('predictions', []):
        key = p['input'][:100]
        pred_lookup[key] = p

    judge_details = judge_results.get('judge_details', [])
    if not judge_details:
        print("\n⚠️  Tidak ada judge details — jalankan dengan --use-judge terlebih dahulu.")
        return

    good, borderline, bad = [], [], []
    for detail in judge_details:
        score = detail.get('judge_rating', 0)
        # Ambil reasoning lengkap dari pred_lookup berdasarkan input_preview
        key   = detail.get('input_preview', '')[:100]
        full  = pred_lookup.get(key, {})

        entry = {
            "judge_score":         score,
            "judge_justification": detail.get('judge_justification', ''),
            "ground_truth_label":  detail.get('gt', ''),
            "predicted_label":     detail.get('pred', ''),
            "correct":             detail.get('gt', '') == detail.get('pred', ''),
            "article_preview":     full.get('input', detail.get('input_preview', ''))[:400],
            "ground_truth_reasoning": full.get('gt_reasoning', '')[:500],
            "model_reasoning":        full.get('pred_reasoning', '')[:500],
            "confidence":          full.get('confidence', None),
        }

        if score >= 4:
            good.append(entry)
        elif score == 3:
            borderline.append(entry)
        else:
            bad.append(entry)

    # Sort: good by score desc, bad by score asc
    good       = sorted(good, key=lambda x: -x['judge_score'])[:n_per_category]
    borderline = sorted(borderline, key=lambda x: -x['judge_score'])[:n_per_category]
    bad        = sorted(bad,        key=lambda x:  x['judge_score'])[:n_per_category]

    output = {
        "_description": (
            "Qualitative reasoning examples, grouped by LLM-as-a-Judge score. "
            "good=score 4-5 (deep & correct), borderline=score 3 (correct but shallow), "
            "bad=score 1-2 (wrong label or incoherent reasoning)."
        ),
        "summary": {
            "total_judged":   len(judge_details),
            "good_count":     len([d for d in judge_details if d.get('judge_rating', 0) >= 4]),
            "borderline_count": len([d for d in judge_details if d.get('judge_rating', 0) == 3]),
            "bad_count":      len([d for d in judge_details if d.get('judge_rating', 0) <= 2]),
            "avg_judge_score": round(judge_results.get('average_rating', 0), 3),
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
    """Plot and save confusion matrix."""
    
    if not HAS_SKLEARN:
        return
    
    print(f"\n📈 Generating confusion matrix...")
    
    labels = ['berita murni', 'native ads']
    cm = confusion_matrix(
        results['ground_truth_labels'],
        results['predicted_labels'],
        labels=labels
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Native Ads Detection', fontsize=14, fontweight='bold')
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to: {output_path}")
    
    return cm


def print_results(results: Dict, bertscore_results: Dict = None, 
                 judge_results: Dict = None, cm: np.ndarray = None):
    """Print comprehensive evaluation results."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*80)
    
    # Overall accuracy
    accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy: {accuracy*100:.2f}% ({results['correct']}/{results['total']})")
    
    # Sklearn metrics
    if HAS_SKLEARN and results['ground_truth_labels']:
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            results['ground_truth_labels'],
            results['predicted_labels'],
            average='weighted'
        )
        
        print(f"   Precision (weighted): {precision*100:.2f}%")
        print(f"   Recall (weighted): {recall*100:.2f}%")
        print(f"   F1-Score (weighted): {f1*100:.2f}%")
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(results['ground_truth_labels'], results['predicted_labels'])
        print(f"   Cohen's Kappa: {kappa:.3f}")
    
    # Per-class metrics
    print(f"\n📈 Per-Class Performance:")
    for label, stats in results['by_class'].items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"   {label}:")
        print(f"      Accuracy: {acc*100:.2f}% ({stats['correct']}/{stats['total']})")
        
        if HAS_SKLEARN:
            # Per-class precision/recall
            class_preds = [p for p in results['predictions'] if p['ground_truth'] == label]
            if class_preds:
                tp = sum(1 for p in class_preds if p['correct'])
                fp = sum(1 for p in results['predictions'] 
                        if p['prediction'] == label and not p['correct'])
                fn = len(class_preds) - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"      Precision: {precision*100:.2f}%")
                print(f"      Recall: {recall*100:.2f}%")
                print(f"      F1-Score: {f1*100:.2f}%")
    
    # NLP Metrics
    if results.get('nlp_metrics'):
        m = results['nlp_metrics']
        print(f"\n📝 NLP Metrics (Reasoning Quality):")
        print(f"   BLEU: {m.get('bleu', 0):.4f}")
        print(f"   METEOR: {m.get('meteor', 0):.4f}")
    
    # Perplexity
    if results.get('perplexities'):
        print(f"   Perplexity: {np.mean(results['perplexities']):.2f}")
    
    # BERTScore results
    if bertscore_results:
        print(f"\n🎯 BERTScore (Reasoning Quality):")
        print(f"   Precision: {bertscore_results['precision']:.3f}")
        print(f"   Recall: {bertscore_results['recall']:.3f}")
        print(f"   F1: {bertscore_results['f1']:.3f}")
    
    # LLM Judge results
    if judge_results and judge_results.get('num_samples_judged', 0) > 0:
        print(f"\n🤖 LLM-as-a-Judge:")
        print(f"   Average Rating: {judge_results['average_rating']:.2f}/5.0")
        print(f"   Samples Judged: {judge_results['num_samples_judged']}")
    
    # Confusion matrix summary
    if cm is not None:
        print(f"\n🔀 Confusion Matrix:")
        print(f"   True Negatives (Berita Murni → Berita Murni): {cm[0][0]}")
        print(f"   False Positives (Berita Murni → Native Ads): {cm[0][1]}")
        print(f"   False Negatives (Native Ads → Berita Murni): {cm[1][0]}")
        print(f"   True Positives (Native Ads → Native Ads): {cm[1][1]}")
    
    # Sample errors
    print(f"\n❌ Sample Errors (First 5):")
    errors = [p for p in results['predictions'] if not p['correct']]
    for i, error in enumerate(errors[:5]):
        print(f"\n   Error {i+1}:")
        print(f"   Input: {error['input']}...")
        print(f"   GT: {error['ground_truth']}, Pred: {error['prediction']} (conf: {error['confidence']:.2f})")
        print(f"   Reasoning: {error['pred_reasoning'][:100]}...")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to fine-tuned model')
    parser.add_argument('--dataset', type=str, default='../data/llm_dataset_finetuning_optimized.json',
                       help='Dataset path')
    parser.add_argument('--num-samples', '--num-sample', type=int, default=100,
                       help='Number of test samples')
    parser.add_argument('--tier', type=str, default='base', choices=['base', 'micro', 'small', 'standard', 'premium'],
                       help='Model tier (base, micro, small, standard, premium) for prompt selection')
    parser.add_argument('--use-rag', action='store_true',
                       help='Use RAG during evaluation')
    parser.add_argument('--vectorstore-dir', type=str, default='data/vectorstore',
                       help='Directory for FAISS vectorstore')
    parser.add_argument('--embedding-model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Embedding model for RAG')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of examples to retrieve for RAG')

    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--lora-path', type=str, default=None,
                       help='Path to LoRA adapters (optional)')
    parser.add_argument('--use-judge', action='store_true',
                       help='Use LLM-as-a-Judge')
    parser.add_argument('--judge-provider', type=str, default='openai',
                       choices=['openai', 'openrouter'],
                       help='Provider for LLM judge (openai, openrouter)')
    parser.add_argument('--judge-model', type=str, default='gpt-4o-mini',
                       help='Model for LLM judge (e.g., openai/gpt-4o for openrouter)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API key for LLM judge (REQUIRED if --use-judge is set)')
    parser.add_argument('--lang', type=str, default=None,
                       help='Filter evaluation by language (e.g. "en", "id")')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU ID to use (0, 1, etc.) or -1 for "auto" (default: 0)')
    parser.add_argument('--allow-rag-mismatch', action='store_true',
                       help='Override the safety check that blocks --use-rag on a non "-rag" model')
    args = parser.parse_args()

    # Extract model name from path
    model_path = Path(args.model)

    # ---- Safety guard (Phase 220): RAG flag must match the model's training ----
    # --use-rag only works on a model fine-tuned with train-time RAG (finetune.py --use-rag),
    # whose dir is named "...-rag". Running --use-rag on a plain model is out-of-distribution
    # and collapses accuracy (the 52% inverted-class failure). Block it loudly.
    model_is_rag = '-rag' in model_path.name.lower()
    if args.use_rag and not model_is_rag and not args.allow_rag_mismatch:
        print("\n" + "="*80)
        print("❌ REFUSING TO RUN: --use-rag on a non-RAG model")
        print("="*80)
        print(f"Model: {args.model}")
        print("This model was NOT trained with train-time RAG (its name lacks '-rag'),")
        print("so injecting retrieved examples is out-of-distribution and WILL collapse accuracy.")
        print("\nFix one of these:")
        print("  1. Retrain first:  python finetune.py --model <m> --use-rag --rag-top-k 4")
        print("     then evaluate ../models/<m>-native-ads-rag")
        print("  2. Drop --use-rag to evaluate this plain model.")
        print("  3. Pass --allow-rag-mismatch to override (NOT recommended).")
        print("="*80 + "\n")
        sys.exit(1)
    if model_is_rag and not args.use_rag:
        print("\n⚠️  WARNING: this is a '-rag' model but --use-rag is NOT set. It was trained")
        print("    WITH embedded examples, so evaluating it plain is also off-distribution.")
        print("    Add --use-rag (and --vectorstore-dir / --top-k) for a fair result.\n")

    model_name = model_path.name.replace('_merged_16bit', '').replace('-', '_')
    
    # Create timestamped output filename
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'eval_{model_name}_{timestamp}_{args.num_samples}samples'
    if args.use_rag:
        output_filename += f'_rag_top{args.top_k}'
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Model Evaluation")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {output_dir / output_filename}")
    print(f"{'='*80}\n")
    
    # Load test set
    test_data = load_test_set(args.dataset, args.num_samples, args.lang)
    
    # Evaluate
    results = evaluate_model(
        args.model,
        test_data,
        lora_path=args.lora_path,
        use_rag=args.use_rag,
        vectorstore_dir=args.vectorstore_dir,
        embedding_model=args.embedding_model,
        gpu_id=args.gpu_id,
        tier=args.tier,
        top_k=args.top_k
    )
    
    # Compute BERTScore
    bertscore_results = compute_bertscore(results)
    
    # Compute NLP Metrics
    nlp_results = compute_nlp_metrics(results)
    results['nlp_metrics'] = nlp_results
    
    # LLM-as-a-Judge
    judge_results = {}
    if args.use_judge:
        judge_results = llm_as_judge(results, args.api_key, args.judge_provider, args.judge_model)

        # Export qualitative reasoning examples (good / borderline / bad)
        examples_path = output_dir / f'{output_filename}_reasoning_examples.json'
        export_reasoning_examples(results, judge_results, str(examples_path))

    # Plot confusion matrix
    cm_path = output_dir / f'{output_filename}_confusion_matrix.png'
    cm = plot_confusion_matrix(results, str(cm_path))
    
    # Print results
    print_results(results, bertscore_results, judge_results, cm)
    
    # Save results
    output_data = {
        'model': args.model,
        'model_name': model_name,
        'timestamp': timestamp,
        'dataset': args.dataset,
        'num_samples': args.num_samples,
        'metrics': {
            'accuracy': results['correct'] / results['total'],
            'total_samples': results['total'],
            'correct': results['correct']
        },
        'bertscore': bertscore_results,
        'nlp_metrics': nlp_results,
        'avg_perplexity': np.mean(results['perplexities']) if results['perplexities'] else 0,
        'llm_judge': judge_results,
        'per_class': {k: {'accuracy': v['correct']/v['total']} 
                     for k, v in results['by_class'].items()},
        'predictions': results['predictions']
    }
    
    # Save JSON results
    json_path = output_dir / f'{output_filename}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Save summary text file
    summary_path = output_dir / f'{output_filename}_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"EVALUATION SUMMARY\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Samples: {args.num_samples}\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy: {output_data['metrics']['accuracy']*100:.2f}%\n")
        f.write(f"  Correct: {output_data['metrics']['correct']}/{output_data['metrics']['total_samples']}\n\n")
        if bertscore_results:
            f.write(f"BERTScore:\n")
            f.write(f"  Precision: {bertscore_results['precision']:.3f}\n")
            f.write(f"  Recall: {bertscore_results['recall']:.3f}\n")
            f.write(f"  F1: {bertscore_results['f1']:.3f}\n\n")
        
        if nlp_results:
            f.write(f"NLP Metrics (Reasoning Quality):\n")
            f.write(f"  BLEU: {nlp_results['bleu']:.4f}\n")
            f.write(f"  METEOR: {nlp_results['meteor']:.4f}\n\n")
            
        if output_data.get('avg_perplexity'):
            f.write(f"Perplexity: {output_data['avg_perplexity']:.2f}\n\n")
            
        f.write(f"Per-Class Accuracy:\n")
        for label, stats in output_data['per_class'].items():
            f.write(f"  {label}: {stats['accuracy']*100:.2f}%\n")
    
    print(f"\n{'='*80}")
    print(f"✅ Results saved to:")
    print(f"  📄 JSON: {json_path}")
    print(f"  📝 Summary: {summary_path}")
    print(f"  📊 Confusion Matrix: {cm_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

