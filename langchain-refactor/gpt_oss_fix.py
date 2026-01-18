#!/usr/bin/env python3
"""
Quick fix for GPT-OSS finetune_gpt_oss.py
Replaces the load_and_format_dataset function with proper title support
"""

# Paste this function to replace the existing load_and_format_dataset in finetune_gpt_oss.py

def load_and_format_dataset(dataset_path: str, tokenizer, max_length: int = 1024):
    """Load, balance, and tokenize dataset."""
    import json
    import random
    import re
    from datasets import Dataset
    
    print(f"\nLoading dataset from: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    if len(data) == 0:
        raise ValueError("Dataset is empty! Check the dataset path.")
    
    # Check if dataset has title field
    has_title = 'title' in data[0] if data else False
    
    # Balance dataset
    print("\n📊 Balancing dataset...")
    native_ads = [d for d in data if 'native ads' in d.get('output', '').lower()]
    berita_murni = [d for d in data if 'native ads' not in d.get('output', '').lower()]
    
    print(f"   Before: Native Ads={len(native_ads)}, Berita Murni={len(berita_murni)}")
    
    if len(native_ads) == 0 or len(berita_murni) == 0:
        raise ValueError(f"Unbalanced dataset! Native ads: {len(native_ads)}, Berita murni: {len(berita_murni)}")
    
    min_count = min(len(native_ads), len(berita_murni))
    
    random.seed(42)
    native_ads_balanced = random.sample(native_ads, min_count)
    berita_murni_balanced = random.sample(berita_murni, min_count)
    
    data = native_ads_balanced + berita_murni_balanced
    random.shuffle(data)
    
    print(f"   After:  Native Ads={len(native_ads_balanced)} (50%), Berita Murni={len(berita_murni_balanced)} (50%)")
    print(f"   Total: {len(data)} balanced samples\n")
    
    if not has_title:
        print("⚠️  WARNING: Dataset does not have 'title' field!")
        print("   Please run: python tools/prepare_finetuning_dataset.py first\n")
    
    # Format for training
    formatted_texts = []
    for sample in data:
        try:
            output_data = json.loads(sample['output'])
            expected_output = json.dumps(output_data, ensure_ascii=False)
        except:
            expected_output = sample['output']
        
        # Use title if available, otherwise extract from input
        if has_title:
            title = sample.get('title', '')
        else:
            sentences = re.split(r'[.!?]\s+', sample['input'])
            title = sentences[0][:100] if sentences else sample['input'][:100]
        
        # Standalone training prompt (avoid import issues)
        prompt = f"""Klasifikasikan berita berikut sebagai "native ads" atau "berita murni".

Native Ads adalah konten yang MENGGABUNGKAN semua ciri berikut:
1. Nada positif/netral (tidak mengkritik subjek)
2. Bahasa persuasif (mengajak/meyakinkan)
3. Mempromosikan produk/brand/instansi
4. Hanya satu sudut pandang (tidak objektif)

Berita Murni:
- Bisa positif/netral/negatif
- Objektif, menyajikan berbagai sudut pandang
- Tidak mempromosikan produk/brand

Judul: {title}
Konten: {sample['input'][:800]}

Output (JSON):
{{"label": "native ads" atau "berita murni", "confidence": 0.0-1.0, "reasoning": "alasan singkat (max 150 karakter)"}}

Klasifikasi:
"""
        text = prompt + expected_output
        formatted_texts.append(text)
    
    # Tokenize the data
    print("Tokenizing dataset...")
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    # Create dataset with tokenized data
    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
    })
    
    split = dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"✓ Training samples: {len(split['train'])}")
    print(f"✓ Evaluation samples: {len(split['test'])}\n")
    
    return split
