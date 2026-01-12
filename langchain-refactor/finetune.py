"""
Fine-Tuning Module for Native Ads Detection
Integrated with LangChain system - uses same prompts and format
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time

# Import prompts from LangChain system
from prompts.classification_prompts import SIMPLE_LOCAL_PROMPT_TEMPLATE

# Model configurations for multi-model support
# Note: GPT-OSS removed due to Unsloth compatibility issues
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


class TrainingMonitor(TrainerCallback):
    """Callback to monitor and log training progress."""
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            self.train_losses.append(logs['loss'])
            self.steps.append(state.global_step)
        if logs and 'eval_loss' in logs:
            self.eval_losses.append(logs['eval_loss'])
    
    def plot_training_curves(self, output_path='training_curves.png'):
        """Plot training and validation loss curves."""
        if not self.train_losses:
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.steps, self.train_losses, label='Training Loss', linewidth=2)
        if self.eval_losses:
            eval_steps = self.steps[::len(self.steps)//len(self.eval_losses)][:len(self.eval_losses)]
            plt.plot(eval_steps, self.eval_losses, label='Validation Loss', 
                    linewidth=2, linestyle='--', marker='o')
        
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Progress - Native Ads Detection', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📈 Training curves saved to: {output_path}")


def load_and_format_dataset(dataset_path: str) -> Dataset:
    """Load dataset and format using EXACT same prompt as inference."""
    
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
    
    # Format with EXACT same prompt as SIMPLE_LOCAL_PROMPT_TEMPLATE
    formatted_data = []
    for sample in data:
        # Parse output to get expected JSON format
        try:
            output_data = json.loads(sample['output'])
            expected_output = json.dumps(output_data, ensure_ascii=False)
        except:
            # If output is not JSON, use as-is
            expected_output = sample['output']
        
        # Use the EXACT prompt template from classification_prompts.py
        text = f"""Klasifikasikan berita berikut sebagai "native ads" atau "berita murni".

Native Ads adalah konten yang MENGGABUNGKAN semua ciri berikut:
1. Nada positif/netral (tidak mengkritik subjek)
2. Bahasa persuasif (mengajak/meyakinkan)
3. Mempromosikan produk/brand/instansi
4. Hanya satu sudut pandang (tidak objektif)

Berita Murni:
- Bisa positif/netral/negatif
- Objektif, menyajikan berbagai sudut pandang
- Tidak mempromosikan produk/brand

Konten:
{sample['input']}

Output (JSON):
{expected_output}"""
        
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune models for Native Ads Detection')
    parser.add_argument('--model', type=str, default='qwen',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Model to fine-tune (qwen, gpt-oss, llama, gemma)')
    parser.add_argument('--dataset', type=str, default='../data/llm_dataset_mixed_json.json',
                       help='Path to training dataset')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: ../models/{model}-native-ads)')
    parser.add_argument('--max-steps', type=int, default=1500,
                       help='Maximum training steps')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of training epochs')
    args = parser.parse_args()
    
    # Get model configuration
    config = MODEL_CONFIGS[args.model]
    
    # Set output directory
    if args.output is None:
        args.output = f"../models/{args.model}-native-ads"
    
    print("="*80)
    print("NATIVE ADS DETECTION - FINE-TUNING")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Description: {config['description']}")
    print(f"Base Model: {config['name']}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"LoRA r: {config['lora_r']}")
    print("="*80 + "\n")
    
    # Check dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {args.dataset}")
        print("Run: python3 ../tools/fix_dataset_format.py first")
        return
    
    # 1. Load base model with Unsloth
    print("[1/5] Loading base model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['name'],
        max_seq_length=config['max_seq_length'],
        dtype=None,
        load_in_4bit=True,
    )
    
    # 2. Add LoRA adapters
    print("[2/5] Adding LoRA adapters...")
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
        use_rslora=False,
        loftq_config=None,
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
    
    # 4. Setup trainer with monitoring
    print("[4/5] Setting up Unsloth SFTTrainer with monitoring...")
    
    # Initialize training monitor
    monitor = TrainingMonitor()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            gradient_accumulation_steps=config['gradient_accumulation'],
            warmup_steps=5,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=config['learning_rate'],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps * 2,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
        ),
        callbacks=[monitor]
    )
    
    # 5. Train
    print("[5/5] Starting training...")
    print("="*80)
    print("⏳ Training in progress...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    try:
        trainer_stats = trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        trainer_stats = None
    
    # Plot training curves
    monitor.plot_training_curves(f"{args.output}/training_curves.png")
    
    # Save model
    print("\n" + "="*80)
    print("Training complete! Saving model...")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save LoRA adapters
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    
    # Save merged model for easier deployment
    print("Saving merged model (16-bit)...")
    merged_path = f"{args.output}_merged_16bit"
    model.save_pretrained_merged(
        merged_path,
        tokenizer,
        save_method="merged_16bit"
    )
    
    print(f"\n✅ Model saved to:")
    print(f"   - LoRA adapters: {args.output}")
    print(f"   - Merged 16-bit: {merged_path}")
    
    if trainer_stats:
        print(f"\n📊 Training Stats:")
        print(f"   Total time: {trainer_stats.metrics['train_runtime']:.2f}s ({trainer_stats.metrics['train_runtime']/3600:.2f} hours)")
        print(f"   Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
        print(f"   Final train loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")
        if 'eval_loss' in trainer_stats.metrics:
            print(f"   Final eval loss: {trainer_stats.metrics['eval_loss']:.4f}")
        
        # Save training stats
        stats_path = f"{args.output}/training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(trainer_stats.metrics, f, indent=2)
        print(f"   Stats saved to: {stats_path}")
    
    print(f"\n🚀 Next Steps:")
    print(f"1. Test: cd .. && python main.py --url YOUR_URL --provider local --model {merged_path}")
    print(f"2. Evaluate: python evaluate_model.py --model {merged_path}")


if __name__ == "__main__":
    main()
