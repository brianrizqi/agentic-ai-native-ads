#!/usr/bin/env python3
"""
Fast OpenRouter Dataset Augmentation with Parallel Processing
Generate detailed reasoning using GPT-4o-mini with 10x speedup
"""

import json
import pandas as pd
import os
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
import time
from tqdm import tqdm

async def generate_detailed_reasoning(client, content: str, label: str, idx: int) -> dict:
    """Generate detailed reasoning using GPT-4o-mini."""
    
    prompt = f"""Analyze this Indonesian news article and provide detailed classification reasoning in JSON format.

**ARTICLE:**
{content}

**GROUND TRUTH LABEL:** {label}

**TASK:**
Analyze based on 4 Native Ads characteristics:

1. **Tone**: Positive/neutral vs Negative/critical (quote examples)
2. **Persuasive Language**: List persuasive words + quote phrases
3. **Brand Promotion**: Name brand/product + how presented
4. **Perspective**: One-sided vs Objective (evidence)

**OUTPUT (JSON only):**
{{
  "label": "{label}",
  "confidence": 0.XX,
  "reasoning": "ANALISIS (Bahasa Indonesia, 200-300 kata):
1. TONE: [jelaskan + quote]
2. PERSUASIF: [list kata + quote]
3. BRAND: [nama + presentasi]
4. PERSPEKTIF: [objektif/one-sided + bukti]
KESIMPULAN: [ringkas]"
}}

Output ONLY valid JSON:"""

    try:
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )
        
        result_text = response.choices[0].message.content
        
        # Parse JSON
        import re
        json_match = re.search(r'\{[^{}]*"label"[^{}]*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                'id': idx,
                'instruction': 'Klasifikasikan artikel berikut sebagai native ads atau berita murni.',
                'input': content,
                'output': json.dumps(result, ensure_ascii=False)
            }
        else:
            raise ValueError("No JSON found")
            
    except Exception as e:
        # Fallback
        return {
            'id': idx,
            'instruction': 'Klasifikasikan artikel berikut sebagai native ads atau berita murni.',
            'input': content,
            'output': json.dumps({
                "label": label,
                "confidence": 0.75,
                "reasoning": f"Error: {str(e)[:100]}"
            }, ensure_ascii=False)
        }


async def process_batch(client, batch_data, semaphore):
    """Process a batch of samples with rate limiting."""
    async with semaphore:
        tasks = [generate_detailed_reasoning(client, row['content'], row['label'], row['idx']) 
                 for row in batch_data]
        return await asyncio.gather(*tasks)


async def main_async(api_key: str, df: pd.DataFrame, max_concurrent: int = 20):
    """Main async processing function."""
    
    # Initialize async client
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    
    # Prepare data
    data = [{'idx': idx, 'content': str(row['content']), 'label': str(row['label'])} 
            for idx, row in df.iterrows()]
    
    # Semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process in batches
    batch_size = 100
    all_results = []
    
    print(f"\n🚀 Processing {len(data)} samples with {max_concurrent} concurrent requests...\n")
    
    with tqdm(total=len(data), desc="Progress") as pbar:
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            results = await process_batch(client, batch, semaphore)
            all_results.extend(results)
            pbar.update(len(batch))
            
            # Small delay between batches
            await asyncio.sleep(0.5)
    
    return all_results


def main():
    print("="*80)
    print("FAST OPENROUTER DATASET AUGMENTATION")
    print("Parallel processing with GPT-4o-mini (10x faster!)")
    print("="*80 + "\n")
    
    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        api_key = input("Enter OpenRouter API key: ").strip()
        if not api_key:
            print("❌ API key required!")
            return
    
    # Load dataset
    input_file = "native_ads_dataset.xlsx"
    if not Path(input_file).exists():
        print(f"❌ {input_file} not found!")
        return
    
    print(f"📊 Loading dataset from {input_file}...")
    df = pd.read_excel(input_file, sheet_name='Clean')
    print(f"✓ Loaded {len(df)} samples\n")
    
    # Estimate
    print(f"⚡ Using 20 concurrent requests")
    print(f"⏱️  Estimated time: ~2-3 hours (vs 29 hours sequential)")
    print(f"💰 Estimated cost: ~$5-10\n")
    
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Run async processing
    start_time = time.time()
    results = asyncio.run(main_async(api_key, df, max_concurrent=20))
    elapsed = time.time() - start_time
    
    # Save results
    output_file = 'data/llm_dataset_detailed.json'
    print(f"\n💾 Saving to {output_file}...")
    
    Path('data').mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Done in {elapsed/3600:.1f} hours!")
    print(f"✅ Saved {len(results)} samples\n")
    print("Next steps:")
    print("1. python tools/fix_dataset_format.py --input data/llm_dataset_detailed.json --output data/llm_dataset_detailed_json.json")
    print("2. cd langchain-refactor && python finetune.py --dataset ../data/llm_dataset_detailed_json.json")


if __name__ == "__main__":
    main()
