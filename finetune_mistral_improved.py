"""
Alternative Fine-Tuning Script - Using Mistral 7B (No Gated Access)
Better alternative to Qwen with no access restrictions
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '0'

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse


def load_dataset(dataset_path: str):
    """Load dataset dari JSON format."""
    print(f"Loading dataset from: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Check balance
    native_count = sum(1 for d in data if 'native ads' in d.get('output', '').lower())
    berita_count = len(data) - native_count
    print(f"\n📊 Dataset Balance:")
    print(f"   Native Ads: {native_count} ({native_count/len(data)*100:.1f}%)")
    print(f"   Berita Murni: {berita_count} ({berita_count/len(data)*100:.1f}%)")
    print(f"   Ratio: {native_count/berita_count:.2f}:1\n")
    
    return data


def prepare_training_data(data, tokenizer, format_type='instruction'):
    """Prepare data untuk training dengan balanced examples."""
    
    def format_instruction(sample):
        """Format sample dengan balanced prompt."""
        if format_type == 'instruction':
            text = f"""<s>[INST] Klasifikasikan konten berita berikut sebagai "native ads" atau "berita murni".

PENTING: Analisis dengan hati-hati. TIDAK semua konten adalah native ads. Berita murni sama umumnya dengan native ads.

Konten:
{sample['input'][:500]}

Klasifikasi: [/INST] {sample['output']}</s>"""
        elif format_type == 'chat':
            messages = sample['messages']
            text = ""
            for msg in messages:
                text += f"{msg['role'].upper()}: {msg['content']}\n\n"
        else:  # qna
            text = f"""Question: {sample['question']}

Context: {sample['context'][:500]}

Answer: {sample['answer']}"""
        
        return text
    
    # Format all samples
    formatted_texts = [format_instruction(sample) for sample in data]
    
    # Tokenize
    print("Tokenizing dataset...")
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask']
    })
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Fine-tune with Mistral (No Gated Access)')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.3',
                       help='Base model (alternatives: microsoft/Phi-3-mini-4k-instruct)')
    parser.add_argument('--dataset', type=str, default='data/llm_dataset_instruction.json')
    parser.add_argument('--output', type=str, default='models/native-ads-mistral-lora')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    args = parser.parse_args()
    
    print("="*80)
    print("IMPROVED FINE-TUNING - Native Ads Detection")
    print("="*80)
    print(f"Base Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print("\n🔧 Improvements:")
    print("   ✅ Lower learning rate (5e-6)")
    print("   ✅ More epochs (5)")
    print("   ✅ Label smoothing (0.1)")
    print("   ✅ Gradient clipping (1.0)")
    print("   ✅ Balanced training prompt")
    print("="*80 + "\n")
    
    if not Path(args.dataset).exists():
        print(f"[ERROR] Dataset not found: {args.dataset}")
        return
    
    # Load tokenizer
    print("[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("[2/6] Loading model (device_map='auto')...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    print(f"✅ Model loaded successfully")
    
    # Apply LoRA
    print("[3/6] Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("[4/6] Loading dataset...")
    data = load_dataset(args.dataset)
    
    format_type = 'instruction'
    if 'messages' in data[0]:
        format_type = 'chat'
    elif 'question' in data[0]:
        format_type = 'qna'
    
    train_dataset = prepare_training_data(data, tokenizer, format_type)
    split = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Training args
    print("[5/6] Setting up training...")
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
        warmup_steps=100,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("[6/6] Starting training...")
    print("="*80)
    print("⏳ Estimated time: 2-3 hours on A100 MIG")
    print("="*80 + "\n")
    
    trainer.train()
    
    # Save
    print("\n" + "="*80)
    print("Training complete! Saving model...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    
    print(f"\n✅ Model saved to: {args.output}")
    print("\n📊 Next: Evaluate the model")
    print(f"python comprehensive_evaluation_full.py --model {args.output}")


if __name__ == "__main__":
    main()
