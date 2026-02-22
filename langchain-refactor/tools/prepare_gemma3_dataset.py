#!/usr/bin/env python3
"""
Prepares the dataset explicitly for Gemma 3 MCQ A/B format.
Extracts title properly, truncates content to 400 chars, maps output to A/B,
and generates the final chat_template text for the trl SFTTrainer.
"""

import json
import argparse
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer

# Phase 8: Reverting to Original Gemma 2 Prompt Style
TRAINING_PROMPT_TEMPLATE = """Judul: {title}
Konten: {content}

===
Berdasarkan teks di atas, apakah artikel tersebut merupakan 'native ads' (iklan terselubung) atau 'berita murni'?
Jawaban:
"""

GEMMA_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ '<start_of_turn>system\\n' + messages[0]['content'] + '<end_of_turn>\\n' }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<start_of_turn>user\\n' + message['content'] + '<end_of_turn>\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<start_of_turn>model\\n' + message['content'] + '<end_of_turn>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<start_of_turn>model\\n' }}"
    "{% endif %}"
)

def prepare_dataset(input_path: str, output_path: str, model_id: str):
    print(f"Loading tokenizer {model_id} for chat template formatting...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Check and set chat template if missing
    if not tokenizer.chat_template:
        print("Tokenizer is missing a chat_template. Setting the default Gemma chat template...")
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE


    print(f"Loading dataset from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")

    processed_data = []
    missing_title_count = 0

    for i, sample in enumerate(data):
        # 1. Title Extraction
        title = sample.get('title')
        if not title:
            missing_title_count += 1
            title = sample.get('input', '')[:100]
        
        # 2. Content Truncation
        content = sample.get('input', '')[:400]

        # 3. Label Mapping (Original text instead of A/B)
        gt_raw = sample.get('output', '').lower()
        if 'native ads' in gt_raw:
            answer = "native ads"
        else:
            answer = "berita murni"

        # 4. Apply Chat Template (Mirroring finetune.py style)
        user_text = TRAINING_PROMPT_TEMPLATE.format(title=title, content=content)
        messages = [
            {"role": "user", "content": user_text}
        ]
        text_prompt_only = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Append answer directly without a strict 'model' turn closure to allow loss calculation on tokens natively
        text = text_prompt_only + answer
        
        # We save it in a generic format that text fields can be loaded from
        processed_data.append({
            "id": sample.get('id', i),
            "text": text
        })

    print(f"\nStats:")
    print(f"- Handled {missing_title_count} empty titles.")
    print(f"- Formatted {len(processed_data)} samples to A/B Chat format.")

    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Prepared dataset saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../../data/llm_dataset_12k_refined.json')
    parser.add_argument('--output', type=str, default='../../data/llm_dataset_gemma3_mcq.json')
    parser.add_argument('--model', type=str, default='google/gemma-3-270m')
    args = parser.parse_args()

    prepare_dataset(args.input, args.output, args.model)

if __name__ == "__main__":
    main()
