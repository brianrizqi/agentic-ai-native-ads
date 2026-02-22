"""
Evaluation Script for Fine-tuned Gemma 3 270M (Hugging Face / LoRA)
Calculates Accuracy, Precision, Recall, F1 and plots Confusion Matrix.
"""

import os
import json
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

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

def load_dataset(dataset_path: str):
    path = Path(dataset_path)
    if not path.exists():
        # Try relative to script location
        path = Path(__file__).parent.parent / "data" / path.name
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_model_output(output_text: str):
    # Phase 8 Label Parsing (Full Text Extraction)
    clean_text = output_text.strip().lower()
    
    if not clean_text:
        return "unknown"
        
    if "native ads" in clean_text:
        return "native ads"
    elif "berita murni" in clean_text:
        return "berita murni"
        
    return "unknown"

def plot_confusion_matrix(y_true, y_pred, save_path):
    labels = ['berita murni', 'native ads']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Gemma 3 270M Fine-tuned')
    plt.savefig(save_path)
    print(f"📈 Confusion matrix saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../models/gemma-3-270m-hf-finetuned', help='Path to LoRA adapters')
    parser.add_argument('--base_model', type=str, default='google/gemma-3-270m')
    parser.add_argument('--dataset', type=str, default='../data/llm_dataset_12k_refined.json')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to test')
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    
    model_path = Path(args.model_path)
    is_merged = not (model_path / "adapter_config.json").exists()
    
    if is_merged:
        # Merged model — load directly in bfloat16
        print(f"🚀 Loading merged model in bfloat16 from: {args.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16, # Kept for explicit precision but using correct var
            token=hf_token
        )
    else:
        # LoRA adapter — load base model in bfloat16
        print(f"🚀 Loading Base Model in bfloat16: {args.base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=hf_token
        )
        print(f"🚀 Loading LoRA Adapters: {args.model_path}...")
        model = PeftModel.from_pretrained(base_model, args.model_path)
        
    # Check and set chat template if missing
    if not hasattr(tokenizer, 'chat_template') or not tokenizer.chat_template:
        print("Tokenizer is missing a chat_template. Setting the default Gemma chat template...")
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE

    model.eval()


    # Load dataset
    data = load_dataset(args.dataset)
    import random
    random.seed(42)
    test_samples = random.sample(data, min(len(data), args.num_samples))

    y_true = []
    y_pred = []
    
    print(f"🧪 Evaluating {len(test_samples)} samples...")
    for i, sample in enumerate(tqdm(test_samples)):
        title = sample.get('title')
        if not title:
            title = sample.get('input', '')[:100]
        
        # Determine ground truth (standardize to 'native ads' or 'berita murni')
        gt_raw = sample.get('output', '').lower()
        if 'native ads' in gt_raw:
            gt_label = 'native ads'
        else:
            gt_label = 'berita murni'
        
        content = sample['input'][:400] # Phase 7 Length Reduction
        user_text = TRAINING_PROMPT_TEMPLATE.format(title=title, content=content)
        
        # Direct prompt matching training
        messages = [{"role": "user", "content": user_text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=150, 
                do_sample=False, # Use greedy decoding for stability
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_label = parse_model_output(response)
        
        if (i < 5):
            print(f"\n--- Debug Sample {i+1} ---")
            print(f"Prompt length (tokens): {inputs.input_ids.shape[1]}")
            print(f"Output shape: {outputs.shape}")
            print(f"Full Decoded Response: |{tokenizer.decode(outputs[0], skip_special_tokens=True)}|")
            print(f"Raw New Tokens Only: |{response}|")
            print(f"Parsed Label: {pred_label}")
            print(f"Ground Truth: {gt_label}")
            print("-" * 20)

        y_true.append(gt_label)
        y_pred.append(pred_label)

    # Filter out unknowns for metrics
    valid_indices = [i for i, p in enumerate(y_pred) if p in ['native ads', 'berita murni']]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    # Metrics
    acc = accuracy_score(y_true_valid, y_pred_valid)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_valid, y_pred_valid, average='binary', pos_label='native ads')

    print("\n" + "="*40)
    print("RESULTS - Gemma 3 270M Fine-tuned")
    print("="*40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Valid Results: {len(y_pred_valid)}/{len(y_pred)}")
    print("="*40)

    # Plot
    plot_confusion_matrix(y_true_valid, y_pred_valid, "gemma3_confusion_matrix.png")

if __name__ == "__main__":
    main()
