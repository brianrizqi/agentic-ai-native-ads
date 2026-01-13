#!/usr/bin/env python3
"""
GPT-OSS Fine-Tuning Script (Separate due to compatibility issues)
Based on Unsloth documentation for gpt-oss
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

# Import prompt template
import sys
sys.path.append(str(Path(__file__).parent))
from prompts.classification_prompts import SIMPLE_LOCAL_PROMPT_TEMPLATE


def load_and_format_dataset(dataset_path: str):
    """Load and balance dataset."""
    
    print(f"\nLoading dataset from: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Balance dataset
    print("\n📊 Balancing dataset...")
    native_ads = [d for d in data if 'native ads' in d.get('output', '').lower()]
    berita_murni = [d for d in data if 'native ads' not in d.get('output', '').lower()]
    
    print(f"   Before: Native Ads={len(native_ads)}, Berita Murni={len(berita_murni)}")
    
    min_count = min(len(native_ads), len(berita_murni))
    
    import random
    random.seed(42)
    native_ads_balanced = random.sample(native_ads, min_count)
    berita_murni_balanced = random.sample(berita_murni, min_count)
    
    data = native_ads_balanced + berita_murni_balanced
    random.shuffle(data)
    
    print(f"   After:  Native Ads={len(native_ads_balanced)} (50%), Berita Murni={len(berita_murni_balanced)} (50%)")
    print(f"   Total: {len(data)} balanced samples\n")
    
    # Format for training
    formatted_data = []
    for sample in data:
        try:
            output_data = json.loads(sample['output'])
            expected_output = json.dumps(output_data, ensure_ascii=False)
        except:
            expected_output = sample['output']
        
        # Use training prompt template from prompts module
        from prompts.classification_prompts import TRAINING_PROMPT_TEMPLATE
        text = TRAINING_PROMPT_TEMPLATE.format(content=sample['input']) + expected_output
        
        formatted_data.append({"text": text})
    
    dataset = Dataset.from_list(formatted_data)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"✓ Training samples: {len(split['train'])}")
    print(f"✓ Evaluation samples: {len(split['test'])}\n")
    
    return split


def main():
    parser = argparse.ArgumentParser(description='Fine-tune GPT-OSS for Native Ads Detection')
    parser.add_argument('--dataset', type=str, default='../data/llm_dataset_mixed_json.json',
                       help='Path to training dataset')
    parser.add_argument('--output', type=str, default='../models/gpt-oss-native-ads',
                       help='Output directory')
    parser.add_argument('--max-steps', type=int, default=1500,
                       help='Maximum training steps')
    parser.add_argument('--max-seq-length', type=int, default=2048,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    print("="*80)
    print("GPT-OSS FINE-TUNING (Experimental)")
    print("="*80)
    print(f"\nModel: GPT-OSS 20B")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Max Seq Length: {args.max_seq_length}")
    print("="*80 + "\n")
    
    print("⚠️  Note: GPT-OSS may have compatibility issues with some Unsloth versions")
    print("   If training fails, this is a known issue with the model architecture.\n")
    
    # Load model - try different model names
    print("🔄 Loading GPT-OSS model...")
    
    model_names_to_try = [
        "unsloth/gpt-oss-20b",  # MXFP4 format
        "unsloth/gpt-oss-20b-bnb-4bit",  # Our custom 4-bit
    ]
    
    model = None
    tokenizer = None
    
    for model_name in model_names_to_try:
        try:
            print(f"   Trying: {model_name}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=args.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            print(f"✓ Successfully loaded: {model_name}\n")
            break
        except Exception as e:
            print(f"   Failed: {e}")
            continue
    
    if model is None:
        print("\n❌ Failed to load GPT-OSS model with any configuration")
        print("   This is a known compatibility issue with GPT-OSS and Unsloth")
        print("   Skipping GPT-OSS training.")
        return
    
    # Add LoRA adapters
    print("🔄 Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print("✓ LoRA adapters added\n")
    
    # Load dataset
    split = load_and_format_dataset(args.dataset)
    
    # Setup trainer
    print("🔄 Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=split['train'],
        eval_dataset=split['test'],
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,
            max_steps=args.max_steps,
            learning_rate=8e-5,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output,
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
    
    try:
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
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print(f"✓ LoRA adapters saved to: {args.output}")
        
        # Save merged model
        merged_path = f"{args.output}_merged_16bit"
        model.save_pretrained_merged(
            merged_path,
            tokenizer,
            save_method="merged_16bit"
        )
        print(f"✓ Merged 16-bit model saved to: {merged_path}\n")
        
        print("🎉 GPT-OSS training successful!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("   This is likely due to GPT-OSS compatibility issues with Unsloth")
        print("   Skipping GPT-OSS model.")


if __name__ == "__main__":
    main()
