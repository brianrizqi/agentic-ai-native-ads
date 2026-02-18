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

# Training prompt template (Must match training!)
TRAINING_PROMPT_TEMPLATE = """Klasifikasikan berita berikut sebagai "native ads" atau "berita murni".

Native Ads adalah konten yang bertujuan untuk PROMOSI atau PERSUASI.
Berita Murni adalah konten INFORMITIF yang objektif.

Judul: {title}
Konten: {content}

Output (JSON):
{{"label": "native ads" atau "berita murni", "confidence": 0.0-1.0, "reasoning": "alasan singkat"}}

Klasifikasi:
"""

def load_dataset(dataset_path: str):
    path = Path(dataset_path)
    if not path.exists():
        # Try relative to script location
        path = Path(__file__).parent.parent / "data" / path.name
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_model_output(output_text: str):
    try:
        # Find JSON block
        start = output_text.find('{')
        end = output_text.rfind('}') + 1
        if start != -1 and end != -1:
            data = json.loads(output_text[start:end])
            return data.get('label', 'unknown').lower()
    except:
        pass
    
    # Fallback keyword matching
    if 'native ads' in output_text.lower():
        return 'native ads'
    elif 'berita murni' in output_text.lower():
        return 'berita murni'
    return 'unknown'

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
    
    print(f"🚀 Loading Base Model: {args.base_model}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token
    )
    
    print(f"🚀 Loading LoRA Adapters: {args.model_path}...")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()

    # Load dataset
    data = load_dataset(args.dataset)
    import random
    random.seed(42)
    test_samples = random.sample(data, min(len(data), args.num_samples))

    y_true = []
    y_pred = []
    
    print(f"🧪 Evaluating {len(test_samples)} samples...")
    for sample in tqdm(test_samples):
        title = sample.get('title', sample.get('input', '')[:100])
        content = sample.get('input', '')[:800]
        
        # Determine ground truth (standardize to 'native ads' or 'berita murni')
        gt_raw = sample.get('output', '').lower()
        if 'native ads' in gt_raw:
            gt_label = 'native ads'
        else:
            gt_label = 'berita murni'
        
        prompt = TRAINING_PROMPT_TEMPLATE.format(title=title, content=content)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.1)
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_label = parse_model_output(response)
        
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
