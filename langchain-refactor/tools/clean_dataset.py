#!/usr/bin/env python3
"""
Clean Training Dataset - Simple Version
Only filters out actual error samples
"""

import json
import argparse

def clean_dataset(input_path, output_path):
    """Remove samples with error reasoning."""
    print(f"Loading dataset from: {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Filter valid samples - only remove actual errors
    clean_data = []
    for sample in data:
        try:
            output = json.loads(sample['output'])
            reasoning = output.get('reasoning', '')
            
            # Skip only if it's an actual error
            if 'Error:' in reasoning or 'No JSON found' in reasoning:
                continue
            
            # Skip if too short
            if len(reasoning.strip()) < 10:
                continue
            
            clean_data.append(sample)
        except:
            continue
    
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
    print(f"\n🚀 Next step:")
    print(f"   python3 finetune.py --dataset {output_path} --output ../models/qwen-native-ads-v4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean training dataset')
    parser.add_argument('--input', type=str,
                       default='../data/llm_dataset_finetuning_optimized.json',
                       help='Input dataset')
    parser.add_argument('--output', type=str,
                       default='../data/llm_dataset_clean.json',
                       help='Output clean dataset')
    
    args = parser.parse_args()
    clean_dataset(args.input, args.output)
