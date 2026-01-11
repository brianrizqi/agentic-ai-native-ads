#!/usr/bin/env python3
"""
Merge Multiple Datasets for Fine-Tuning
Combines llm_dataset_instruction.json with llm_dataset_detailed.json
"""

import json
import argparse
from pathlib import Path
import random

def load_dataset(file_path):
    """Load dataset from JSON file."""
    print(f"Loading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  ✓ Loaded {len(data)} samples")
    return data

def merge_datasets(datasets, shuffle=True):
    """Merge multiple datasets."""
    merged = []
    
    for dataset in datasets:
        merged.extend(dataset)
    
    if shuffle:
        random.seed(42)
        random.shuffle(merged)
    
    # Re-index
    for idx, sample in enumerate(merged):
        sample['id'] = idx
    
    return merged

def analyze_dataset(data):
    """Analyze dataset composition."""
    total = len(data)
    
    # Count by label
    native_count = 0
    berita_count = 0
    
    for sample in data:
        output = sample.get('output', '')
        if 'native ads' in output.lower():
            native_count += 1
        else:
            berita_count += 1
    
    print(f"\n📊 Dataset Analysis:")
    print(f"   Total samples: {total}")
    print(f"   Native Ads: {native_count} ({native_count/total*100:.1f}%)")
    print(f"   Berita Murni: {berita_count} ({berita_count/total*100:.1f}%)")
    
    # Check reasoning length
    reasoning_lengths = []
    for sample in data[:100]:  # Sample first 100
        output = sample.get('output', '')
        reasoning_lengths.append(len(output))
    
    avg_length = sum(reasoning_lengths) / len(reasoning_lengths)
    print(f"   Avg output length: {avg_length:.0f} chars")

def main():
    parser = argparse.ArgumentParser(description='Merge multiple datasets')
    parser.add_argument('--inputs', nargs='+', required=True,
                       help='Input dataset files to merge')
    parser.add_argument('--output', type=str, required=True,
                       help='Output merged dataset file')
    parser.add_argument('--no-shuffle', action='store_true',
                       help='Do not shuffle merged dataset')
    args = parser.parse_args()
    
    print("="*80)
    print("DATASET MERGER")
    print("="*80 + "\n")
    
    # Load all datasets
    datasets = []
    for input_file in args.inputs:
        if not Path(input_file).exists():
            print(f"❌ File not found: {input_file}")
            return
        datasets.append(load_dataset(input_file))
    
    # Merge
    print(f"\n🔄 Merging {len(datasets)} datasets...")
    merged = merge_datasets(datasets, shuffle=not args.no_shuffle)
    
    # Analyze
    analyze_dataset(merged)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to {args.output}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Merged dataset saved: {len(merged)} samples\n")
    print("Next steps:")
    print(f"1. python tools/fix_dataset_format.py --input {args.output} --output data/merged_dataset_json.json")
    print(f"2. cd langchain-refactor && python finetune.py --dataset ../data/merged_dataset_json.json")

if __name__ == "__main__":
    main()
