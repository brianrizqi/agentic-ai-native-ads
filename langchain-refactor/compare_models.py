#!/usr/bin/env python3
"""
Multi-Model Comparison Script
Compare performance of all fine-tuned models for research paper
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, cohen_kappa_score
from agents.classification_agent import ClassificationAgent
import time
from datetime import datetime

# Model paths (4 models including experimental GPT-OSS)
MODELS = {
    "Qwen 2.5 14B": "../models/qwen-native-ads_merged_16bit",
    "GPT-OSS 20B": "../models/gpt-oss-native-ads_merged_16bit",
    "Llama 3.1 8B": "../models/llama-native-ads_merged_16bit",
    "Gemma 2 9B": "../models/gemma-native-ads_merged_16bit"
}


def evaluate_model(model_name: str, model_path: str, test_samples: list, verbose: bool = False):
    """Evaluate a single model."""
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")
    
    if not Path(model_path).exists():
        print(f"⚠️  Model not found: {model_path}")
        print(f"   Skipping {model_name}")
        return None
    
    # Initialize agent
    try:
        agent = ClassificationAgent(
            model_name=model_path,
            provider="local"
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Run evaluation
    predictions = []
    ground_truth = []
    inference_times = []
    error_count = 0
    
    print(f"Running evaluation on {len(test_samples)} samples...")
    
    for i, sample in enumerate(test_samples):
        if verbose and i % 20 == 0:
            print(f"  Progress: {i}/{len(test_samples)}")
        
        # Get ground truth
        gt_output = json.loads(sample['output'])
        gt_label = gt_output['label']
        ground_truth.append(gt_label)
        
        # Get prediction
        start_time = time.time()
        try:
            result = agent.classify(
                title="",
                summary="",
                content=sample['input'],
                context=""
            )
            predictions.append(result['label'])
            inference_times.append(time.time() - start_time)
        except Exception as e:
            if verbose:
                print(f"  Error on sample {i}: {e}")
            error_count += 1
            predictions.append("error")
            inference_times.append(0)
    
    print(f"  Completed: {len(test_samples)} samples, {error_count} errors")
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='weighted', zero_division=0
    )
    kappa = cohen_kappa_score(ground_truth, predictions)
    
    # Per-class metrics
    labels = sorted(set(ground_truth))
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        ground_truth, predictions, labels=labels, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=labels)
    
    # Inference speed
    avg_inference_time = np.mean([t for t in inference_times if t > 0])
    
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "num_samples": len(test_samples),
        "overall_metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "cohens_kappa": float(kappa)
        },
        "per_class_metrics": {
            label: {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1_score": float(f1_per_class[i]),
                "support": int(support[i])
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "inference_speed": {
            "avg_time_seconds": float(avg_inference_time),
            "samples_per_second": float(1 / avg_inference_time) if avg_inference_time > 0 else 0
        }
    }
    
    print(f"\n📊 Results for {model_name}:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   Avg Inference: {avg_inference_time:.2f}s")
    
    return results


def create_comparison_plots(all_results: list, output_dir: Path):
    """Create comparison visualizations."""
    
    print(f"\n📊 Creating comparison plots...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    model_names = [r['model_name'] for r in all_results]
    accuracies = [r['overall_metrics']['accuracy'] for r in all_results]
    f1_scores = [r['overall_metrics']['f1_score'] for r in all_results]
    inference_times = [r['inference_speed']['avg_time_seconds'] for r in all_results]
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. F1-Score Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Inference Speed Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, inference_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel('Avg Inference Time (seconds)', fontsize=12)
    plt.title('Model Inference Speed Comparison', fontsize=14, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_speed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Confusion Matrices (2x2 grid for 4 models)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, result in enumerate(all_results):
        cm = np.array(result['confusion_matrix'])
        labels = result['labels']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   ax=axes[idx])
        axes[idx].set_title(f"{result['model_name']}\nAccuracy: {result['overall_metrics']['accuracy']:.3f}",
                           fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plots saved to {output_dir}/")


def create_comparison_table(all_results: list, output_dir: Path):
    """Create comparison table."""
    
    # Create DataFrame
    data = []
    for result in all_results:
        data.append({
            'Model': result['model_name'],
            'Accuracy': f"{result['overall_metrics']['accuracy']:.4f}",
            'Precision': f"{result['overall_metrics']['precision']:.4f}",
            'Recall': f"{result['overall_metrics']['recall']:.4f}",
            'F1-Score': f"{result['overall_metrics']['f1_score']:.4f}",
            "Cohen's κ": f"{result['overall_metrics']['cohens_kappa']:.4f}",
            'Avg Inference (s)': f"{result['inference_speed']['avg_time_seconds']:.2f}"
        })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_path = output_dir / 'model_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Comparison table saved to {csv_path}")
    
    # Print table
    print(f"\n{'='*80}")
    print("MODEL COMPARISON TABLE")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))
    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Compare all fine-tuned models')
    parser.add_argument('--dataset', type=str, default='../data/llm_dataset_mixed_json.json',
                       help='Test dataset path')
    parser.add_argument('--num-samples', type=int, default=200,
                       help='Number of samples to evaluate')
    parser.add_argument('--output', type=str, default='../results/model_comparison',
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MULTI-MODEL COMPARISON")
    print("="*80)
    print(f"\nDataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {args.output}")
    print(f"\nModels to compare:")
    for name, path in MODELS.items():
        status = "✓" if Path(path).exists() else "✗"
        print(f"  {status} {name}: {path}")
    print("="*80 + "\n")
    
    # Load test dataset
    with open(args.dataset, 'r') as f:
        data = json.load(f)
    
    import random
    random.seed(42)
    test_samples = random.sample(data, min(args.num_samples, len(data)))
    
    # Evaluate all models
    all_results = []
    for model_name, model_path in MODELS.items():
        result = evaluate_model(model_name, model_path, test_samples, args.verbose)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("\n❌ No models were successfully evaluated!")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_file = output_dir / 'comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_samples': args.num_samples,
            'results': all_results
        }, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")
    
    # Create visualizations
    create_comparison_plots(all_results, output_dir)
    
    # Create comparison table
    create_comparison_table(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print("🎉 COMPARISON COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved in: {output_dir}/")
    print(f"  - comparison_results.json")
    print(f"  - model_comparison.csv")
    print(f"  - accuracy_comparison.png")
    print(f"  - f1_comparison.png")
    print(f"  - inference_speed_comparison.png")
    print(f"  - confusion_matrices.png")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
