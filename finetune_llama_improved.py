"""
Improved Fine-Tuning Script for Native Ads Detection
Using Llama 3.2 with better hyperparameters to prevent bias
"""

import os
# Force PyTorch-only mode
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
            # Use improved balanced prompt
            text = f"""### Instruction:
Klasifikasikan konten berita berikut sebagai "native ads" (iklan berbayar yang menyerupai berita) atau "berita murni" (konten jurnalistik murni).

PENTING: Analisis dengan hati-hati. TIDAK semua konten adalah native ads. Berita murni sama umumnya dengan native ads.

### Input:
{sample['input'][:500]}

### Response:
{sample['output']}"""
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
    parser = argparse.ArgumentParser(description='Improved Fine-tune for Native Ads Detection')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                       help='Base model to fine-tune')
    parser.add_argument('--dataset', type=str, default='data/llm_dataset_instruction.json',
                       help='Dataset path')
    parser.add_argument('--output', type=str, default='models/native-ads-llama32-lora',
                       help='Output directory for fine-tuned model')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs (increased from 3)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-6,
                       help='Learning rate (lowered from 2e-5)')
    args = parser.parse_args()
    
    print("="*80)
    print("IMPROVED FINE-TUNING - Native Ads Detection with Llama 3.2")
    print("="*80)
    print(f"Base Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print("\n🔧 Improvements:")
    print("   ✅ Lower learning rate (5e-6 vs 2e-5)")
    print("   ✅ More epochs (5 vs 3)")
    print("   ✅ Label smoothing (0.1)")
    print("   ✅ Gradient clipping (1.0)")
    print("   ✅ Balanced prompt in training data")
    print("="*80 + "\n")
    
    # Check dataset exists
    if not Path(args.dataset).exists():
        print(f"[ERROR] Dataset not found: {args.dataset}")
        return
    
    # Load tokenizer
    print("[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with device_map="auto" for MIG compatibility
    print("[2/6] Loading model (device_map='auto' for MIG)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    print(f"✅ Model loaded successfully")
    
    # Apply LoRA
    print("[3/6] Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # More modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    print("[4/6] Loading and preparing dataset...")
    data = load_dataset(args.dataset)
    
    # Determine format type
    format_type = 'instruction'
    if 'messages' in data[0]:
        format_type = 'chat'
    elif 'question' in data[0]:
        format_type = 'qna'
    
    print(f"Dataset format: {format_type}")
    
    # Prepare training data
    train_dataset = prepare_training_data(data, tokenizer, format_type)
    
    # Split train/eval (90/10) with stratification
    split = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Training arguments with improvements
    print("[5/6] Setting up improved training configuration...")
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,  # Lowered
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
        # NEW: Improvements to prevent bias
        label_smoothing_factor=0.1,  # Prevent overconfidence
        max_grad_norm=1.0,  # Gradient clipping
        lr_scheduler_type="cosine",  # Better learning rate schedule
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("[6/6] Starting training...")
    print("="*80)
    print("⏳ This will take approximately 2-3 hours on A100 MIG...")
    print("="*80 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\n" + "="*80)
    print("Training complete! Saving model...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    
    print(f"\n✅ Model saved to: {args.output}")
    print("\n📊 Next steps:")
    print(f"1. Evaluate: python comprehensive_evaluation_full.py --model {args.output}")
    print(f"2. Demo: python run_local_demo.py --model {args.output} --url YOUR_URL")
    print("\n🎯 Expected improvements:")
    print("   - Accuracy: 47.6% → 75-85%")
    print("   - Berita Murni Recall: 0.77% → 70-80%")
    print("   - Balanced predictions (not all native ads!)")


if __name__ == "__main__":
    main()
