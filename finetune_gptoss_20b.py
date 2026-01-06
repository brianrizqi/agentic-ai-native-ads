"""
GPT-OSS 20B Fine-Tuning with Aggressive Memory Optimization
Using 8-bit quantization + gradient checkpointing to fit in MIG
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
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
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
    """Prepare data untuk training."""
    
    def format_instruction(sample):
        """Format sample dengan balanced prompt."""
        if format_type == 'instruction':
            text = f"""### Instruction:
Klasifikasikan konten berita berikut sebagai "native ads" atau "berita murni".

PENTING: Analisis dengan hati-hati. TIDAK semua konten adalah native ads.

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
    parser = argparse.ArgumentParser(description='GPT-OSS 20B with 8-bit Quantization')
    parser.add_argument('--model', type=str, default='openai/gpt-oss-20b')
    parser.add_argument('--dataset', type=str, default='data/llm_dataset_instruction.json')
    parser.add_argument('--output', type=str, default='models/native-ads-gptoss-lora')
    parser.add_argument('--epochs', type=int, default=3)  # Reduced for large model
    parser.add_argument('--batch-size', type=int, default=1)  # Must be 1 for 20B
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    args = parser.parse_args()
    
    print("="*80)
    print("GPT-OSS 20B Fine-Tuning with AGGRESSIVE Memory Optimization")
    print("="*80)
    print(f"Base Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print("\n🔧 Memory Optimizations:")
    print("   ✅ 8-bit quantization (load_in_8bit)")
    print("   ✅ Gradient checkpointing")
    print("   ✅ Batch size = 1")
    print("   ✅ Gradient accumulation = 8")
    print("   ✅ LoRA rank = 4 (minimal)")
    print("="*80 + "\n")
    
    if not Path(args.dataset).exists():
        print(f"[ERROR] Dataset not found: {args.dataset}")
        return
    
    # Load tokenizer
    print("[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 8-bit quantization
    print("[2/6] Loading model with 8-bit quantization...")
    print("⚠️  This will take several minutes and use ~20GB memory...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    print(f"✅ Model loaded with 8-bit quantization")
    
    # Apply LoRA with minimal rank
    print("[3/6] Applying LoRA (minimal rank for memory)...")
    lora_config = LoraConfig(
        r=4,  # Minimal rank for 20B model
        lora_alpha=8,
        target_modules=["c_attn", "c_proj"],
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
    
    # Training args with aggressive memory saving
    print("[5/6] Setting up training...")
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,  # Effective batch = 8
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=400,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        warmup_steps=100,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,  # Critical for memory
        optim="adamw_8bit",  # 8-bit optimizer
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
    print("⏳ Estimated time: 3-4 hours (20B model is SLOW)")
    print("⚠️  If OOM error occurs, the model is too big for your MIG slice")
    print("="*80 + "\n")
    
    try:
        trainer.train()
        
        # Save
        print("\n" + "="*80)
        print("Training complete! Saving model...")
        trainer.save_model(args.output)
        tokenizer.save_pretrained(args.output)
        
        print(f"\n✅ Model saved to: {args.output}")
        print("\n📊 Next: Evaluate the model")
        print(f"python comprehensive_evaluation_full.py --model {args.output}")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "NVML" in str(e):
            print("\n" + "="*80)
            print("❌ OUT OF MEMORY ERROR")
            print("="*80)
            print("GPT-OSS 20B is too large for your MIG configuration.")
            print("\n💡 Recommended alternatives:")
            print("1. GPT-2 Medium (345M) - Safe and fast")
            print("2. GPT-Neo 1.3B - Good balance")
            print("3. Phi-2 (2.7B) - Modern and efficient")
            print("\nRun with smaller model:")
            print("python finetune_mistral_improved.py --model openai-community/gpt2-medium")
        else:
            raise e


if __name__ == "__main__":
    main()
