#!/usr/bin/env python3
"""
Simple OpenRouter-based Dataset Augmentation
Generate detailed reasoning for native ads classification using GPT-4o-mini
"""

import json
import pandas as pd
import os
import sys
from pathlib import Path
from openai import OpenAI
import time

def generate_detailed_reasoning(client, content: str, label: str) -> dict:
    """Generate detailed reasoning using GPT-4o-mini."""
    
    prompt = f"""Analyze this Indonesian news article and provide detailed classification reasoning in JSON format.

**ARTICLE:**
{content}

**GROUND TRUTH LABEL:** {label}

**TASK:**
Analyze based on 4 Native Ads characteristics and provide evidence:

1. **Tone**: Positive/neutral (no criticism) vs Negative/critical
   - Quote specific sentences showing tone
   
2. **Persuasive Language**: Words that convince/persuade readers
   - List persuasive words found (e.g., "terbaik", "wajib", "solusi")
   - Quote at least 2 persuasive phrases
   
3. **Brand Promotion**: Promotes product/brand/institution
   - Name the brand/product mentioned
   - Explain how it's presented
   
4. **Perspective**: One-sided vs Objective/balanced
   - Is there criticism or only praise?
   - Are multiple viewpoints presented?

**OUTPUT FORMAT (JSON):**
{{
  "label": "{label}",
  "confidence": 0.XX,
  "reasoning": "ANALISIS DETAIL (Bahasa Indonesia, 200-300 kata):

1. TONE & NADA:
[Jelaskan tone dengan quote spesifik]

2. BAHASA PERSUASIF:
[List kata persuasif + quote]

3. PROMOSI BRAND:
[Brand/produk + cara presentasi]

4. PERSPEKTIF:
[Objektif atau one-sided + bukti]

KESIMPULAN:
[Ringkas kenapa {label}]"
}}

**IMPORTANT:**
- Reasoning in Bahasa Indonesia
- Include specific quotes (use \\" for quotes in JSON)
- 200-300 words
- Confidence: 0.75-0.95

Output ONLY valid JSON:"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        
        result_text = response.choices[0].message.content
        
        # Try to parse JSON
        import re
        json_match = re.search(r'\{[^{}]*"label"[^{}]*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            # Fallback
            return {
                "label": label,
                "confidence": 0.75,
                "reasoning": result_text[:500]
            }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "label": label,
            "confidence": 0.75,
            "reasoning": f"Error generating reasoning: {str(e)}"
        }


def main():
    print("="*80)
    print("OPENROUTER DATASET AUGMENTATION")
    print("Generate detailed reasoning with GPT-4o-mini")
    print("="*80 + "\n")
    
    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        api_key = input("Enter OpenRouter API key: ").strip()
        if not api_key:
            print("❌ API key required!")
            return
    
    # Initialize client
    print("✓ Initializing OpenRouter client...")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    
    # Load dataset
    input_file = "native_ads_dataset.xlsx"
    if not Path(input_file).exists():
        print(f"❌ {input_file} not found!")
        return
    
    print(f"📊 Loading dataset from {input_file}...")
    df = pd.read_excel(input_file, sheet_name='Clean')
    print(f"✓ Loaded {len(df)} samples\n")
    
    # Confirm
    print(f"💰 Estimated cost: ~$5-10 for {len(df)} samples")
    print(f"⏱️  Estimated time: ~30-60 minutes\n")
    
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Process samples
    results = []
    print(f"\n🔄 Processing {len(df)} samples...\n")
    
    for idx, row in df.iterrows():
        content = str(row['content'])
        label = str(row['label'])
        
        if idx % 100 == 0:
            print(f"Progress: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")
        
        # Generate reasoning
        result = generate_detailed_reasoning(client, content, label)
        
        # Create dataset entry
        output_json = json.dumps(result, ensure_ascii=False)
        
        results.append({
            'id': idx,
            'instruction': 'Klasifikasikan artikel berikut sebagai native ads atau berita murni.',
            'input': content,
            'output': output_json
        })
        
        # Rate limiting
        time.sleep(0.1)
    
    # Save results
    output_file = 'data/llm_dataset_detailed.json'
    print(f"\n💾 Saving to {output_file}...")
    
    Path('data').mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Done! Saved {len(results)} samples\n")
    print("Next steps:")
    print("1. python tools/fix_dataset_format.py --input data/llm_dataset_detailed.json --output data/llm_dataset_detailed_json.json")
    print("2. cd langchain-refactor && python finetune.py --dataset ../data/llm_dataset_detailed_json.json")


if __name__ == "__main__":
    main()
