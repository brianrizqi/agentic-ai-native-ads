#!/usr/bin/env python3
"""
Clean Training Dataset
Filters out samples with error reasoning for better training quality
"""

import json
import argparse
from pathlib import Path

def is_valid_sample(sample):
    """Check if sample has valid reasoning (no errors)."""
    try:
        output = json.loads(sample['output'])
        reasoning = output.get('reasoning', '')
        
        # Skip samples with error indicators
        error_indicators = [
            'Error:',
            'No JSON found',
            'invalid',
            'ANALISIS (Bahasa Indonesia',  # Too long format
            'Extracted from incomplete',
            'Empty or invalid'
        ]
        
        for indicator in error_indicators:
            if indicator in reasoning:
                return False
        
        # Check if reasoning is too short (likely error)
        if len(reasoning.strip()) < 10:
            return False
        
        return True
    except:
        return False

def clean_dataset(input_path, output_path):
    """Remove samples with error reasoning."""
    print(f"Loading dataset from: {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Filter valid samples
    clean_data = [s for s in data if is_valid_sample(s)]
    
    skipped = len(data) - len(clean_data)
    print(f"Valid samples: {len(clean_data)}")
    print(f"Skipped (errors): {skipped} ({skipped/len(data)*100:.1f}%)")
    
    # Check label balance
    labels = {}
    for sample in clean_data:
        output = json.loads(sample['output'])
        label = output['label']
        labels[label] = labels.get(label, 0) + 1
    
    print(f"\nLabel distribution:")
    for label, count in labels.items():
        print(f"  {label}: {count} ({count/len(clean_data)*100:.1f}%)")
    
    # Save clean dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Clean dataset saved to: {output_path}")
    
    return {
        'total': len(data),
        'clean': len(clean_data),
        'skipped': skipped,
        'labels': labels
    }

def main():
    parser = argparse.ArgumentParser(description='Clean training dataset')
    parser.add_argument('--input', type=str,
                       default='../data/llm_dataset_finetuning_optimized.json',
                       help='Input dataset')
    parser.add_argument('--output', type=str,
                       default='../data/llm_dataset_clean.json',
                       help='Output clean dataset')
    
    args = parser.parse_args()
    
    stats = clean_dataset(args.input, args.output)
    
    print(f"\n🚀 Next step:")
    print(f"   python3 finetune.py --dataset {args.output} --output ../models/qwen-native-ads-v4")

if __name__ == "__main__":
    main()
