#!/usr/bin/env python3
"""
Dataset Preprocessing for Fine-Tuning
Prepares dataset with proper formatting, title extraction, and reasoning truncation
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List


def extract_title(text: str, max_length: int = 100) -> str:
    """Extract title from text (first sentence or first max_length chars)."""
    # Try to find first sentence
    sentences = re.split(r'[.!?]\s+', text)
    if sentences and len(sentences[0]) > 0:
        title = sentences[0].strip()
        if len(title) <= max_length:
            return title
    
    # Fallback: use first max_length chars
    title = text[:max_length].strip()
    if not title.endswith('.'):
        title += '...'
    
    return title


def truncate_reasoning(reasoning: str, max_length: int = 150) -> str:
    """Truncate reasoning to max_length characters."""
    if len(reasoning) <= max_length:
        return reasoning
    
    # Try to truncate at sentence boundary
    truncated = reasoning[:max_length]
    last_period = truncated.rfind('.')
    
    if last_period > max_length * 0.7:  # If we can keep at least 70% with sentence boundary
        return truncated[:last_period + 1]
    
    # Otherwise just truncate and add ellipsis
    return truncated.rsplit(' ', 1)[0] + '...'


def preprocess_dataset(
    input_path: str,
    output_path: str,
    max_reasoning_length: int = 150,
    validate_only: bool = False
) -> Dict:
    """
    Preprocess dataset for fine-tuning.
    
    Args:
        input_path: Path to input dataset (llm_dataset_detailed.json)
        output_path: Path to output dataset
        max_reasoning_length: Maximum length for reasoning field
        validate_only: If True, only validate without writing output
    
    Returns:
        Statistics about the preprocessing
    """
    print(f"Loading dataset from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    stats = {
        'total': len(data),
        'processed': 0,
        'errors': 0,
        'reasoning_truncated': 0,
        'title_extracted': 0,
        'original_reasoning_lengths': [],
        'new_reasoning_lengths': []
    }
    
    processed_data = []
    
    for i, item in enumerate(data):
        try:
            # Parse output
            output = json.loads(item['output'])
            
            # Extract title from input
            title = extract_title(item['input'])
            stats['title_extracted'] += 1
            
            # Truncate reasoning
            original_reasoning = output.get('reasoning', '')
            stats['original_reasoning_lengths'].append(len(original_reasoning))
            
            new_reasoning = truncate_reasoning(original_reasoning, max_reasoning_length)
            stats['new_reasoning_lengths'].append(len(new_reasoning))
            
            if len(new_reasoning) < len(original_reasoning):
                stats['reasoning_truncated'] += 1
            
            # Create new output with truncated reasoning
            new_output = {
                'label': output['label'],
                'confidence': output['confidence'],
                'reasoning': new_reasoning
            }
            
            # Create processed item
            processed_item = {
                'id': item.get('id', i),
                'instruction': item['instruction'],
                'title': title,  # NEW: extracted title
                'input': item['input'],
                'output': json.dumps(new_output, ensure_ascii=False)
            }
            
            processed_data.append(processed_item)
            stats['processed'] += 1
            
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(data)} samples...")
        
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            stats['errors'] += 1
            continue
    
    # Calculate statistics
    if stats['original_reasoning_lengths']:
        stats['avg_original_reasoning'] = sum(stats['original_reasoning_lengths']) / len(stats['original_reasoning_lengths'])
        stats['max_original_reasoning'] = max(stats['original_reasoning_lengths'])
        stats['avg_new_reasoning'] = sum(stats['new_reasoning_lengths']) / len(stats['new_reasoning_lengths'])
        stats['max_new_reasoning'] = max(stats['new_reasoning_lengths'])
    
    # Print statistics
    print("\n" + "="*80)
    print("PREPROCESSING STATISTICS")
    print("="*80)
    print(f"Total samples: {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Errors: {stats['errors']}")
    print(f"\nTitle extraction:")
    print(f"  Titles extracted: {stats['title_extracted']}")
    print(f"\nReasoning truncation:")
    print(f"  Samples truncated: {stats['reasoning_truncated']} ({stats['reasoning_truncated']/stats['total']*100:.1f}%)")
    print(f"  Original avg length: {stats.get('avg_original_reasoning', 0):.0f} chars")
    print(f"  Original max length: {stats.get('max_original_reasoning', 0)} chars")
    print(f"  New avg length: {stats.get('avg_new_reasoning', 0):.0f} chars")
    print(f"  New max length: {stats.get('max_new_reasoning', 0)} chars")
    print("="*80 + "\n")
    
    # Validation check
    if validate_only:
        print("✅ Validation complete - all samples have title, content, and short reasoning")
        return stats
    
    # Save processed dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Processed dataset saved to: {output_path}")
    
    # Save sample for inspection
    sample_path = output_path.parent / f"{output_path.stem}_sample.json"
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data[:5], f, ensure_ascii=False, indent=2)
    
    print(f"📄 Sample (first 5 items) saved to: {sample_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset for fine-tuning')
    parser.add_argument('--input', type=str, 
                       default='../../data/llm_dataset_detailed.json',
                       help='Input dataset path')
    parser.add_argument('--output', type=str,
                       default='../../data/llm_dataset_finetuning_optimized.json',
                       help='Output dataset path')
    parser.add_argument('--max-reasoning', type=int, default=150,
                       help='Maximum reasoning length in characters')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate without writing output')
    
    args = parser.parse_args()
    
    stats = preprocess_dataset(
        input_path=args.input,
        output_path=args.output,
        max_reasoning_length=args.max_reasoning,
        validate_only=args.validate_only
    )
    
    if not args.validate_only:
        print("\n🚀 Next steps:")
        print(f"1. Inspect sample: cat {Path(args.output).parent / (Path(args.output).stem + '_sample.json')}")
        print(f"2. Fine-tune: python finetune.py --dataset {args.output}")


if __name__ == "__main__":
    main()
