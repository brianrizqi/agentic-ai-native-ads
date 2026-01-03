"""
Fine-Tuning Script for Native Ads Detection
Compatible with A100 MIG mode using device_map="auto"
"""

import os
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
    return data

def prepare_training_data(data, tokenizer, format_type='instruction'):
    """Prepare data untuk training."""
    
    def format_instruction(sample):
        """Format sample sebagai instruction."""
        if format_type == 'instruction':
            text = f"""### Instruction:
{sample['instruction']}

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
    parser = argparse.ArgumentParser(description='Fine-tune LLM for Native Ads Detection')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                       help='Base model to fine-tune')
    parser.add_argument('--dataset', type=str, default='data/llm_dataset_instruction.json',
                       help='Dataset path')
    parser.add_argument('--output', type=str, default='models/native-ads-detector',
                       help='Output directory for fine-tuned model')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--use-lora', action='store_true',
                       help='Use LoRA for efficient fine-tuning')
    args = parser.parse_args()
    
    print("="*80)
    print("FINE-TUNING SCRIPT - Native Ads Detection")
    print("="*80)
    print(f"Base Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"LoRA: {args.use_lora}")
    print("="*80 + "\n")
    
    # Check dataset exists
    if not Path(args.dataset).exists():
        print(f"[ERROR] Dataset not found: {args.dataset}")
        print("Run: python tools/dataset_converter.py first")
        return
    
    # Load tokenizer
    print("[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with device_map="auto" for MIG compatibility
    print("[2/6] Loading model (using device_map='auto' for MIG compatibility)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",  # Auto handle MIG
        torch_dtype=torch.float16,  # Use FP16 for efficiency
        trust_remote_code=True
    )
    
    print(f"Model loaded on device: {model.device}")
    
    # Apply LoRA if requested
    if args.use_lora:
        print("[3/6] Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("[3/6] Using full fine-tuning (no LoRA)")
    
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
    
    # Split train/eval (90/10)
    split = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Training arguments
    print("[5/6] Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=True,  # Use mixed precision
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb
        warmup_steps=100,
        weight_decay=0.01,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
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
    trainer.train()
    
    # Save final model
    print("\n" + "="*80)
    print("Training complete! Saving model...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    
    print(f"\n✅ Model saved to: {args.output}")
    print("\nTo use the fine-tuned model:")
    print(f"python run_local_demo.py --model {args.output} --url YOUR_URL")

if __name__ == "__main__":
    main()
