"""
Fine-Tuning Script using Unsloth (2x faster, 50% less memory)
Optimized for Native Ads Detection with Expert Classification Rules
"""

import json
import argparse
from pathlib import Path
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments


def load_and_format_dataset(dataset_path: str):
    """Load dataset and format for instruction tuning."""
    
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
    
    # Format with expert rules prompt (matching classification_prompts.py)
    formatted_data = []
    for sample in data:
        # Use EXACT same format as simple_local_prompt
        text = f"""### Instruction:
Klasifikasikan berita berikut sebagai "native ads" atau "berita murni".

Native Ads adalah konten yang MENGGABUNGKAN semua ciri berikut:
1. Nada positif/netral (tidak mengkritik subjek)
2. Bahasa persuasif (mengajak/meyakinkan)
3. Mempromosikan produk/brand/instansi
4. Hanya satu sudut pandang (tidak objektif)

Berita Murni:
- Bisa positif/netral/negatif
- Objektif, menyajikan berbagai sudut pandang
- Tidak mempromosikan produk/brand

### Input:
{sample['input'][:500]}

### Response:
{sample['output']}"""
        
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)


def main():
    parser = argparse.ArgumentParser(description='Unsloth Fine-tuning for Native Ads Detection')
    parser.add_argument('--model', type=str, default='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
                       help='''Base model options (8B+ recommended):
                       - unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit (Recommended)
                       - unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit (12B, very good)
                       - unsloth/gemma-2-9b-it-bnb-4bit (9B, Google)
                       - unsloth/Qwen2.5-7B-Instruct-bnb-4bit (7B, multilingual)
                       ''')
    parser.add_argument('--dataset', type=str, default='data/llm_dataset_instruction.json',
                       help='Dataset path')
    parser.add_argument('--output', type=str, default='models/native-ads-llama8b-lora',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Per device batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--quantization', type=str, default='4bit', choices=['4bit', '8bit'],
                       help='Quantization level (4bit uses less memory, 8bit slightly better quality)')
    args = parser.parse_args()
    
    print("="*80)
    print("UNSLOTH FINE-TUNING - Native Ads Detection")
    print("="*80)
    print(f"Base Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print("\n🚀 Unsloth Benefits:")
    print("   ✅ 2x faster training")
    print("   ✅ 50% less memory usage")
    print("   ✅ Zero quality degradation")
    print("="*80 + "\n")
    
    # Check dataset exists
    if not Path(args.dataset).exists():
        print(f"[ERROR] Dataset not found: {args.dataset}")
        return
    
    # 1. Load model with Unsloth optimization
    print("[1/5] Loading model with Unsloth optimization...")
    print(f"Quantization: {args.quantization}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=None,  # Auto-detect
        load_in_4bit=(args.quantization == '4bit'),  # Use 4-bit or 8-bit
    )
    
    # 2. Add LoRA adapters (following Unsloth GPT-OSS recommendations)
    print("[2/5] Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,  # Unsloth recommends 8 for GPT-OSS (can use 16, 32, 64 for larger models)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Unsloth optimized (0 is faster)
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=3407,
        use_rslora=False,  # Rank stabilized LoRA
        loftq_config=None,  # LoftQ config
    )
    
    # 3. Load and prepare dataset
    print("[3/5] Loading and formatting dataset...")
    dataset = load_and_format_dataset(args.dataset)
    
    # Split train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # 4. Setup trainer
    print("[4/5] Setting up Unsloth SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=4,  # Unsloth recommendation
            warmup_steps=5,  # Unsloth uses 5 for GPT-OSS
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,  # More frequent logging for monitoring
            optim="adamw_8bit",  # Unsloth's optimized optimizer
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to="none",  # Disable wandb/tensorboard unless needed
        ),
    )
    
    # 5. Train
    print("[5/5] Starting training...")
    print("="*80)
    print("⏳ Training in progress (Unsloth accelerated)...")
    print("="*80 + "\n")
    
    trainer_stats = trainer.train()
    
    # Save model
    print("\n" + "="*80)
    print("Training complete! Saving model...")
    
    # Save LoRA adapters
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    
    # Also save merged model (optional, for easier deployment)
    print("Saving merged model (16-bit)...")
    model.save_pretrained_merged(
        f"{args.output}_merged_16bit",
        tokenizer,
        save_method="merged_16bit"
    )
    
    print(f"\n✅ Model saved to:")
    print(f"   - LoRA adapters: {args.output}")
    print(f"   - Merged 16-bit: {args.output}_merged_16bit")
    
    print("\n📊 Training Stats:")
    print(f"   Total time: {trainer_stats.metrics['train_runtime']:.2f}s")
    print(f"   Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
    
    print("\n🚀 Next Steps:")
    print(f"1. Test: python main.py --url YOUR_URL --provider local --model {args.output}_merged_16bit")
    print(f"2. Evaluate: python comprehensive_evaluation_full.py --model {args.output}_merged_16bit")


if __name__ == "__main__":
    main()
