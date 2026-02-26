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


def load_test_set(dataset_path: str, num_samples: int = 100) -> List[Dict]:
    """Load test samples from dataset. Shuffles with seed to ensure diversity."""
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Shuffle with fixed seed to be reproducible but diverse
    import random
    random.seed(42)
    random.shuffle(data)
    
    # Take samples
    test_data = data[:num_samples]
    
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


def evaluate_model(model_path: str, test_data: List[Dict], lora_path: Optional[str] = None) -> Dict:
    """Evaluate model on test set with comprehensive metrics."""
    
    print(f"Initializing model from: {model_path}")
    if lora_path:
        print(f"Using LoRA adapters from: {lora_path}")
    
    agent = ClassificationAgent(
        provider='local', 
        model_name=model_path,
        lora_path=lora_path
    )
    
    results = {
        'correct': 0,
        'total': 0,
        'by_class': defaultdict(lambda: {'correct': 0, 'total': 0, 'predictions': []}),
        'confusion_matrix': defaultdict(int),
        'predictions': [],
        'ground_truth_labels': [],
        'predicted_labels': [],
        'reasoning_texts': {'ground_truth': [], 'predicted': []}
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
            pred = agent.classify(sample['input'], title=title)
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
        
        # Store reasoning for BERTScore
        results['reasoning_texts']['ground_truth'].append(gt_reasoning)
        results['reasoning_texts']['predicted'].append(pred_reasoning)
        
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
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_data)} samples...")
    
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
    
    # Final safety check: ensure all are strings
    gt_texts = [str(t) for t in gt_texts]
    pred_texts = [str(t) for t in pred_texts]
    
    # Run BERTScore on CPU to avoid conflicts with classification model on GPU
    # Do NOT call torch.cuda.empty_cache() here - GPU may be in error state
    try:
        P, R, F1 = bert_score(pred_texts, gt_texts, lang='id', verbose=False, device='cpu')
    except Exception as e:
        print(f"⚠️  BERTScore failed: {e}")
        return {}

    
    return {
        'precision': float(P.mean()),
        'recall': float(R.mean()),
        'f1': float(F1.mean())
    }



def llm_as_judge(results: Dict, api_key: str = None, provider: str = "openai", model: str = "gpt-4o-mini") -> Dict:
    """Use an LLM (OpenAI or OpenRouter) to judge prediction quality, prioritizing errors."""
    
    if not api_key:
        print(f"⚠️  Skipping LLM-as-a-Judge: API Key missing. Please provide via --api-key.")
        return {}
    
    final_api_key = api_key
    
    print(f"\n🤖 Running LLM-as-a-Judge evaluation (Provider: {provider}, Model: {model})...")
    
    if provider == "openrouter":
        llm = ChatOpenAI(
            model=model,
            openai_api_key=final_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
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
    
    sample_size = min(20, len(all_predictions))
    # Mix: up to 15 incorrect, 5 correct
    judge_samples = (incorrect[:15] + correct)[:sample_size]
    
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
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of test samples')
    parser.add_argument('--output-dir', type=str, default='eval_results',
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
    args = parser.parse_args()
    
    # Extract model name from path
    model_path = Path(args.model)
    model_name = model_path.name.replace('_merged_16bit', '').replace('-', '_')
    
    # Create timestamped output filename
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'eval_{model_name}_{timestamp}'
    
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
    test_data = load_test_set(args.dataset, args.num_samples)
    
    # Evaluate
    results = evaluate_model(args.model, test_data, lora_path=args.lora_path)
    
    # Compute BERTScore
    bertscore_results = compute_bertscore(results)
    
    # LLM-as-a-Judge
    judge_results = {}
    if args.use_judge:
        judge_results = llm_as_judge(results, args.api_key, args.judge_provider, args.judge_model)
    
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

