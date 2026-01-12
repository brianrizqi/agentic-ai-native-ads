#!/usr/bin/env python3
"""
Multi-Model Fine-Tuning Script
Supports: Qwen 2.5 14B, GPT-OSS 20B, Llama 3.1 8B, Gemma 2 9B
"""

import argparse
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import time
from datetime import datetime

# Model configurations
MODEL_CONFIGS = {
    "qwen": {
        "name": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        "max_seq_length": 2048,
        "lora_r": 8,
        "lora_alpha": 16,
        "batch_size": 1,
        "gradient_accumulation": 4,
        "learning_rate": 1e-4,
        "description": "Qwen 2.5 14B - Best multilingual performance"
    },
    "gpt-oss": {
        "name": "unsloth/gpt-oss-20b-bnb-4bit",
        "max_seq_length": 2048,
        "lora_r": 8,
        "lora_alpha": 16,
        "batch_size": 1,
        "gradient_accumulation": 4,
        "learning_rate": 8e-5,
        "description": "GPT-OSS 20B - Largest model"
    },
    "llama": {
        "name": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "max_seq_length": 2048,
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 2,
        "gradient_accumulation": 2,
        "learning_rate": 1e-4,
        "description": "Llama 3.1 8B - Balanced speed/quality"
    },
    "gemma": {
        "name": "unsloth/gemma-2-9b-it-bnb-4bit",
        "max_seq_length": 2048,
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 2,
        "gradient_accumulation": 2,
        "learning_rate": 1e-4,
        "description": "Gemma 2 9B - Google's efficient model"
    }
}


def load_and_format_dataset(dataset_path: str):
    """Load and format dataset for training."""
    print(f"\n📊 Loading dataset from {dataset_path}...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} samples")
    
    # Format for training
    formatted_data = []
    for sample in data:
        try:
            output_data = json.loads(sample['output'])
            expected_output = json.dumps(output_data, ensure_ascii=False)
        except:
            expected_output = sample['output']
        
        text = f"""Klasifikasikan berita berikut sebagai "native ads" atau "berita murni".

Native Ads adalah konten yang MENGGABUNGKAN semua ciri berikut:
1. Nada positif/netral (tidak mengkritik subjek)
2. Bahasa persuasif (mengajak/meyakinkan)
3. Mempromosikan produk/brand/instansi
4. Hanya satu sudut pandang (tidak objektif)

Konten:
{sample['input']}

Output (JSON):
{expected_output}"""
        
        formatted_data.append({"text": text})
    
    # Create dataset and split
    dataset = Dataset.from_list(formatted_data)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"✓ Training samples: {len(split['train'])}")
    print(f"✓ Evaluation samples: {len(split['test'])}")
    
    return split


def train_model(model_key: str, dataset_path: str, output_dir: str, max_steps: int = 1500):
    """Train a specific model."""
    
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Choose from: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_key]
    
    print("="*80)
    print(f"FINE-TUNING: {config['description']}")
    print("="*80)
    print(f"Model: {config['name']}")
    print(f"Output: {output_dir}")
    print(f"Max Steps: {max_steps}")
    print("="*80 + "\n")
    
    # Load model
    print("🔄 Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['name'],
        max_seq_length=config['max_seq_length'],
        dtype=None,
        load_in_4bit=True,
    )
    print("✓ Model loaded\n")
    
    # Add LoRA adapters
    print("🔄 Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora_r'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config['lora_alpha'],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print("✓ LoRA adapters added\n")
    
    # Load dataset
    split = load_and_format_dataset(dataset_path)
    
    # Setup trainer
    print("\n🔄 Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split['train'],
        eval_dataset=split['test'],
        dataset_text_field="text",
        max_seq_length=config['max_seq_length'],
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            gradient_accumulation_steps=config['gradient_accumulation'],
            warmup_steps=5,
            num_train_epochs=1,
            max_steps=max_steps,
            learning_rate=config['learning_rate'],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
        ),
    )
    print("✓ Trainer configured\n")
    
    # Train
    print("🚀 Starting training...")
    print("="*80)
    start_time = time.time()
    
    trainer_stats = trainer.train()
    
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("✅ Training complete!")
    print(f"⏱️  Total time: {elapsed/3600:.2f} hours")
    print(f"📊 Final train loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")
    print(f"📊 Final eval loss: {trainer_stats.metrics.get('eval_loss', 'N/A')}")
    print("="*80 + "\n")
    
    # Save model
    print("💾 Saving model...")
    
    # Save LoRA adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ LoRA adapters saved to: {output_dir}")
    
    # Save merged model (16-bit)
    merged_path = f"{output_dir}_merged_16bit"
    model.save_pretrained_merged(
        merged_path,
        tokenizer,
        save_method="merged_16bit"
    )
    print(f"✓ Merged 16-bit model saved to: {merged_path}")
    
    # Save training stats
    stats = {
        "model": model_key,
        "model_name": config['name'],
        "description": config['description'],
        "training_time_hours": elapsed / 3600,
        "max_steps": max_steps,
        "final_train_loss": float(trainer_stats.metrics.get('train_loss', 0)),
        "final_eval_loss": float(trainer_stats.metrics.get('eval_loss', 0)),
        "config": config,
        "timestamp": datetime.now().isoformat()
    }
    
    stats_file = Path(output_dir) / "training_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Training stats saved to: {stats_file}\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Multi-Model Fine-Tuning for Native Ads Detection')
    parser.add_argument('--model', type=str, required=True,
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Model to fine-tune')
    parser.add_argument('--dataset', type=str, default='data/llm_dataset_mixed_json.json',
                       help='Path to training dataset')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: models/{model}-native-ads)')
    parser.add_argument('--max-steps', type=int, default=1500,
                       help='Maximum training steps')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output is None:
        args.output = f"models/{args.model}-native-ads"
    
    # Show model info
    config = MODEL_CONFIGS[args.model]
    print("\n" + "="*80)
    print("MULTI-MODEL FINE-TUNING")
    print("="*80)
    print(f"\nSelected Model: {args.model}")
    print(f"Description: {config['description']}")
    print(f"Base Model: {config['name']}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Max Steps: {args.max_steps}")
    print("\n" + "="*80 + "\n")
    
    # Confirm
    confirm = input("Continue with training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Train
    stats = train_model(args.model, args.dataset, args.output, args.max_steps)
    
    print("\n" + "="*80)
    print("🎉 ALL DONE!")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Output: {args.output}")
    print(f"Merged: {args.output}_merged_16bit")
    print(f"Training time: {stats['training_time_hours']:.2f} hours")
    print("\nNext steps:")
    print(f"1. Evaluate: python scripts/compare_models.py --model {args.output}_merged_16bit")
    print(f"2. Test: python langchain-refactor/main.py --provider local --model {args.output}_merged_16bit --url <URL>")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
