"""
Alternative Fine-Tuning Script - MIG Compatible
Uses manual training loop instead of Trainer to avoid MIG issues
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import json
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, Dataset


class NativeAdsDataset(Dataset):
    """Custom dataset for native ads detection."""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Format text
        if 'instruction' in sample:
            text = f"{sample['instruction']}\n\n{sample['input']}\n\n{sample['output']}"
        elif 'question' in sample:
            text = f"{sample['question']}\n\n{sample['context']}\n\n{sample['answer']}"
        else:
            text = str(sample)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


def main():
    parser = argparse.ArgumentParser(description='Fine-tune with MIG compatibility')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--dataset', type=str, default='data/llm_dataset_instruction.json')
    parser.add_argument('--output', type=str, default='models/native-ads-mistral-lora')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max-steps', type=int, default=1000)
    args = parser.parse_args()
    
    print("="*80)
    print("FINE-TUNING (MIG Compatible Mode)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Max Steps: {args.max_steps}")
    print("="*80 + "\n")
    
    # Check dataset
    if not Path(args.dataset).exists():
        print(f"[ERROR] Dataset not found: {args.dataset}")
        return
    
    # Load dataset
    print("[1/6] Loading dataset...")
    with open(args.dataset, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Limit to max_steps * batch_size samples
    max_samples = args.max_steps * args.batch_size
    if len(data) > max_samples:
        data = data[:max_samples]
    
    print(f"Using {len(data)} samples")
    
    # Load tokenizer
    print("[2/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with CPU offloading for MIG
    print("[3/6] Loading model (CPU offload mode for MIG)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_8bit=True,  # 8-bit quantization
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Apply LoRA
    print("[4/6] Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    print("[5/6] Preparing dataset...")
    train_dataset = NativeAdsDataset(data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    print("[6/6] Training...")
    print("="*80)
    
    model.train()
    global_step = 0
    total_loss = 0
    
    try:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                if global_step >= args.max_steps:
                    break
                
                # Move to device
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Update progress
                avg_loss = total_loss / global_step
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # Log every 100 steps
                if global_step % 100 == 0:
                    print(f"\nStep {global_step}: loss={avg_loss:.4f}")
            
            if global_step >= args.max_steps:
                break
    
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user")
    
    except Exception as e:
        print(f"\n\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save model
    print("\n" + "="*80)
    print("Saving model...")
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    
    print(f"✅ Model saved to: {args.output}")
    print(f"Total steps: {global_step}")
    print(f"Final loss: {total_loss / global_step:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
