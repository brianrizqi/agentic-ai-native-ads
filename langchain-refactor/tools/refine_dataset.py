import json
import argparse
from pathlib import Path
from typing import List, Dict
import re

def refine_reasoning(sample: Dict) -> Dict:
    """
    Refine reasoning for specific cases to reduce model bias.
    Specifically targeting 'berita murni' samples that mention brands but are currently generic.
    """
    input_text = sample.get('input', '').lower()
    output_str = sample.get('output', '')
    
    try:
        output_data = json.loads(output_str)
        label = output_data.get('label', '')
        reasoning = output_data.get('reasoning', '')
    except json.JSONDecodeError:
        return sample

    # Generic reasoning often found in the dataset
    generic_patterns = [
        "artikel bersifat netral dan objektif",
        "menyajikan informasi tanpa promosi produk/brand",
        "tidak ada ajakan persuatif",
        "bersifat informatif"
    ]
    
    is_generic = any(p in reasoning.lower() for p in generic_patterns)
    
    if label == "berita murni" and is_generic:
        # Check for brand mentions or corporate news characteristics
        corporate_keywords = ["pt ", "tbk", "perusahaan", "saham", "investasi", "laba", "pendapatan", "direktur", "ceo"]
        brand_mentions = [k for k in corporate_keywords if k in input_text]
        
        if brand_mentions:
            # This is a 'Hard Negative' - News mentioning brands/money but not promotional
            refined_reasoning = f"Meskipun menyebutkan {', '.join(brand_mentions[:2])}, artikel ini bersifat informatif mengenai kinerja/peristiwa korporasi dan tidak mengandung unsur persuasi atau promosi produk."
            output_data['reasoning'] = refined_reasoning
            sample['output'] = json.dumps(output_data, ensure_ascii=False)
            sample['is_hard_negative'] = True
            
    return sample

def main():
    parser = argparse.ArgumentParser(description='Refine dataset reasoning and identify hard negatives.')
    parser.add_argument('--input', type=str, default='../data/llm_dataset_12k_simple.json', help='Input dataset path')
    parser.add_argument('--output', type=str, default='../data/llm_dataset_12k_refined.json', help='Output dataset path')
    parser.add_argument('--check-only', action='store_true', help='Only check without writing')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        # Try local path if called from tools/
        input_path = Path(args.input.replace('../', ''))
        if not input_path.exists():
            print(f"Error: Dataset not found at {args.input}")
            return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing {len(data)} samples...")
    
    refined_count = 0
    hard_negatives = 0
    
    refined_data = []
    removed_empty = 0
    for sample in data:
        # Check for empty input
        input_text = sample.get('input', '').strip()
        if not input_text:
            removed_empty += 1
            continue
            
        original_output = sample.get('output', '')
        refined_sample = refine_reasoning(sample)
        
        if refined_sample.get('is_hard_negative'):
            hard_negatives += 1
            
        if refined_sample.get('output') != original_output:
            refined_count += 1
            
        refined_data.append(refined_sample)

    print(f"Refined {refined_count} samples.")
    print(f"Identified {hard_negatives} hard negatives.")
    print(f"Removed {removed_empty} samples with empty input.")
    
    if not args.check_only:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(refined_data, f, ensure_ascii=False, indent=2)
        print(f"Saved refined dataset to {args.output}")

if __name__ == "__main__":
    main()
