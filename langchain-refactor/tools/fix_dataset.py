#!/usr/bin/env python3
"""
Fix Dataset - Keep 3000 samples but fix error reasoning
Instead of removing error samples, replace their reasoning with valid ones
"""

import json
import argparse
from collections import Counter

def generate_fallback_reasoning(label, content_snippet):
    """Generate simple but valid reasoning based on label."""
    if label == "native ads":
        return "Artikel memiliki nada positif dan mempromosikan produk/brand tertentu."
    else:
        return "Artikel bersifat netral dan objektif, menyajikan informasi tanpa promosi."

def fix_dataset(input_path, output_path):
    """Fix error reasoning while keeping all 3000 samples."""
    print(f"Loading dataset from: {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    fixed_count = 0
    error_types = Counter()
    
    for sample in data:
        try:
            output = json.loads(sample['output'])
            reasoning = output.get('reasoning', '')
            label = output.get('label', '')
            
            # Check if reasoning has errors
            has_error = False
            error_type = None
            
            if 'Error:' in reasoning or 'No JSON found' in reasoning:
                has_error = True
                error_type = 'JSON Error'
            elif len(reasoning.strip()) < 10:
                has_error = True
                error_type = 'Too Short'
            elif reasoning.startswith('ANALISIS (Bahasa Indonesia') and len(reasoning) > 200:
                has_error = True
                error_type = 'Too Long (ANALISIS format)'
            
            # Fix if needed
            if has_error:
                # Get content snippet for context
                content_snippet = sample['input'][:100]
                
                # Generate fallback reasoning
                new_reasoning = generate_fallback_reasoning(label, content_snippet)
                
                # Update output
                output['reasoning'] = new_reasoning
                sample['output'] = json.dumps(output, ensure_ascii=False)
                
                fixed_count += 1
                error_types[error_type] += 1
        
        except Exception as e:
            print(f"Warning: Could not process sample: {e}")
            continue
    
    print(f"\nFixed samples: {fixed_count}/{len(data)} ({fixed_count/len(data)*100:.1f}%)")
    print(f"\nError types fixed:")
    for error_type, count in error_types.most_common():
        print(f"  {error_type}: {count}")
    
    # Check label balance
    labels = {}
    for sample in data:
        output = json.loads(sample['output'])
        label = output['label']
        labels[label] = labels.get(label, 0) + 1
    
    print(f"\nLabel distribution:")
    for label, count in labels.items():
        print(f"  {label}: {count} ({count/len(data)*100:.1f}%)")
    
    # Save fixed dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Fixed dataset saved to: {output_path}")
    print(f"   Total samples: {len(data)} (same as input)")
    print(f"\n🚀 Next step:")
    print(f"   python3 finetune.py --dataset {output_path} --output ../models/qwen-native-ads-v5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fix dataset errors while keeping all samples')
    parser.add_argument('--input', type=str,
                       default='../data/llm_dataset_finetuning_optimized.json',
                       help='Input dataset')
    parser.add_argument('--output', type=str,
                       default='../data/llm_dataset_fixed.json',
                       help='Output fixed dataset')
    
    args = parser.parse_args()
    fix_dataset(args.input, args.output)
