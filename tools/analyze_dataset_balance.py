"""
Dataset Balance Analyzer
Analyzes class distribution in training dataset to identify imbalance issues.
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_dataset(dataset_path: str):
    """Analyze class distribution in dataset."""
    
    print("="*80)
    print("DATASET BALANCE ANALYSIS")
    print("="*80)
    print(f"Dataset: {dataset_path}\n")
    
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_samples = len(data)
    print(f"Total Samples: {total_samples}\n")
    
    # Detect format and extract labels
    labels = []
    
    for sample in data:
        if 'output' in sample:
            # Instruction format
            output = sample['output'].lower()
            if 'native ads' in output or 'native advertising' in output:
                labels.append('native ads')
            else:
                labels.append('berita murni')
        elif 'answer' in sample:
            # QnA format
            answer = sample['answer'].lower()
            if 'native ads' in answer or 'native advertising' in answer:
                labels.append('native ads')
            else:
                labels.append('berita murni')
        elif 'messages' in sample:
            # Chat format
            last_msg = sample['messages'][-1]['content'].lower()
            if 'native ads' in last_msg or 'native advertising' in last_msg:
                labels.append('native ads')
            else:
                labels.append('berita murni')
    
    # Count distribution
    counter = Counter(labels)
    
    native_count = counter.get('native ads', 0)
    berita_count = counter.get('berita murni', 0)
    
    native_pct = (native_count / total_samples) * 100
    berita_pct = (berita_count / total_samples) * 100
    
    # Print results
    print("--- Class Distribution ---")
    print(f"Native Ads:    {native_count:5d} ({native_pct:5.1f}%)")
    print(f"Berita Murni:  {berita_count:5d} ({berita_pct:5.1f}%)")
    print(f"{'':15} {'─'*20}")
    print(f"Total:         {total_samples:5d} (100.0%)\n")
    
    # Calculate imbalance ratio
    if berita_count > 0:
        ratio = native_count / berita_count
        print(f"Imbalance Ratio: {ratio:.2f}:1 (native ads : berita murni)")
    else:
        ratio = float('inf')
        print("⚠️  WARNING: No 'berita murni' samples found!")
    
    # Imbalance severity
    print("\n--- Imbalance Severity ---")
    if abs(ratio - 1.0) < 0.1:
        severity = "✅ BALANCED"
        recommendation = "Dataset is well-balanced. No action needed."
    elif abs(ratio - 1.0) < 0.3:
        severity = "⚠️  SLIGHTLY IMBALANCED"
        recommendation = "Consider using class weighting during training."
    elif abs(ratio - 1.0) < 0.5:
        severity = "🟠 MODERATELY IMBALANCED"
        recommendation = "Use class weighting + balanced sampling."
    else:
        severity = "🔴 SEVERELY IMBALANCED"
        recommendation = "CRITICAL: Use focal loss + oversampling minority class."
    
    print(f"Status: {severity}")
    print(f"Recommendation: {recommendation}\n")
    
    # Calculate suggested class weights
    if native_count > 0 and berita_count > 0:
        total = native_count + berita_count
        weight_native = total / (2 * native_count)
        weight_berita = total / (2 * berita_count)
        
        print("--- Suggested Class Weights ---")
        print(f"Native Ads Weight:   {weight_native:.4f}")
        print(f"Berita Murni Weight: {weight_berita:.4f}")
        print("\nAdd to training script:")
        print(f"class_weights = torch.tensor([{weight_native:.4f}, {weight_berita:.4f}])")
    
    # Plot distribution
    plot_distribution(counter, dataset_path)
    
    print("\n" + "="*80)
    
    return {
        'total': total_samples,
        'native_ads': native_count,
        'berita_murni': berita_count,
        'ratio': ratio,
        'severity': severity
    }


def plot_distribution(counter: Counter, dataset_path: str):
    """Plot class distribution."""
    
    labels = list(counter.keys())
    counts = list(counter.values())
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    colors = ['#ff6b6b', '#4ecdc4']
    bars = plt.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/sum(counts)*100:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Class Distribution in Training Dataset', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = Path(dataset_path).parent / 'class_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Distribution plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset class balance')
    parser.add_argument('--dataset', type=str, default='data/llm_dataset_instruction.json',
                       help='Path to dataset JSON file')
    args = parser.parse_args()
    
    if not Path(args.dataset).exists():
        print(f"❌ ERROR: Dataset not found: {args.dataset}")
        return
    
    analyze_dataset(args.dataset)


if __name__ == "__main__":
    main()
