"""
Dataset Format Converter - Fix Training/Inference Mismatch
Converts dataset from Markdown format to JSON format matching inference prompt
"""

import json
import re
from pathlib import Path
import argparse


def parse_markdown_output(output: str) -> dict:
    """Parse markdown format output to structured dict."""
    
    # Extract label
    label_match = re.search(r'\*\*Klasifikasi\*\*:\s*(native ads|berita murni)', output, re.IGNORECASE)
    label = label_match.group(1) if label_match else "unknown"
    
    # Extract confidence - try multiple patterns
    confidence = 0.7  # default
    
    # Pattern 1: **Confidence**: 0.95
    conf_match = re.search(r'\*\*Confidence\*\*:\s*([\d.]+)', output, re.IGNORECASE)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
        except:
            pass
    else:
        # Pattern 2: **Confidence**: High/Medium/Low
        conf_text_match = re.search(r'\*\*Confidence\*\*:\s*(High|Medium|Low)', output, re.IGNORECASE)
        if conf_text_match:
            conf_str = conf_text_match.group(1).lower()
            if conf_str == 'high':
                confidence = 0.85
            elif conf_str == 'medium':
                confidence = 0.7
            elif conf_str == 'low':
                confidence = 0.55
        else:
            # Pattern 3: confidence: 0.XX or confidence = 0.XX
            conf_alt = re.search(r'confidence[:\s=]+([\d.]+)', output, re.IGNORECASE)
            if conf_alt:
                try:
                    confidence = float(conf_alt.group(1))
                except:
                    pass
    
    # Ensure confidence is in valid range
    confidence = max(0.0, min(1.0, confidence))
    
    # Extract reasoning/analysis
    analysis_match = re.search(r'\*\*Analisis\*\*:\s*(.+?)(?:\n\*\*|$)', output, re.DOTALL)
    if not analysis_match:
        # Try alternative patterns
        analysis_match = re.search(r'(?:Reasoning|Alasan|Penjelasan):\s*(.+?)(?:\n|$)', output, re.DOTALL | re.IGNORECASE)
    
    reasoning = analysis_match.group(1).strip() if analysis_match else f"Konten diklasifikasikan sebagai {label}"
    
    return {
        "label": label,
        "confidence": round(confidence, 2),
        "reasoning": reasoning[:200]  # Truncate to 200 chars
    }


def convert_dataset(input_path: str, output_path: str):
    """Convert dataset from Markdown to JSON format."""
    
    print(f"Loading dataset from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    converted = []
    for i, sample in enumerate(data):
        # Parse markdown output to structured format
        parsed = parse_markdown_output(sample['output'])
        
        # Create new sample with JSON output
        new_sample = {
            'id': sample.get('id', i),
            'instruction': sample.get('instruction', ''),
            'input': sample['input'],
            'output': json.dumps(parsed, ensure_ascii=False)
        }
        
        converted.append(new_sample)
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(data)} samples...")
    
    # Check balance
    native_count = sum(1 for s in converted if 'native ads' in s['output'].lower())
    berita_count = len(converted) - native_count
    
    print(f"\n📊 Dataset Balance:")
    print(f"   Native Ads: {native_count} ({native_count/len(converted)*100:.1f}%)")
    print(f"   Berita Murni: {berita_count} ({berita_count/len(converted)*100:.1f}%)")
    
    # Check confidence distribution
    confidences = [json.loads(s['output'])['confidence'] for s in converted]
    from collections import Counter
    conf_dist = Counter([round(c, 1) for c in confidences])
    
    print(f"\n📈 Confidence Distribution:")
    for conf in sorted(conf_dist.keys(), reverse=True):
        count = conf_dist[conf]
        print(f"   {conf:.1f}: {count} samples ({count/len(converted)*100:.1f}%)")

    
    # Save converted dataset
    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Conversion complete! Saved {len(converted)} samples")
    
    # Show sample
    print(f"\n📝 Sample converted output:")
    print(converted[0]['output'])


def main():
    parser = argparse.ArgumentParser(description='Convert dataset format')
    parser.add_argument('--input', type=str, default='data/llm_dataset_instruction.json',
                       help='Input dataset path')
    parser.add_argument('--output', type=str, default='data/llm_dataset_json_format.json',
                       help='Output dataset path')
    args = parser.parse_args()
    
    convert_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
