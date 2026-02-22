"""
Fine-Tuning Script for Gemma 3 270M (Pure Hugging Face)
Uses transformers, PEFT, and TRL.
"""

import os
import json
import torch
import argparse
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Phase 7: Recency Bias Fix (Inverted Prompt & Drastic Length Reduction)
TRAINING_PROMPT_TEMPLATE = """Judul: {title}
Konten: {content}

===
Berdasarkan teks di atas, apakah artikel tersebut merupakan 'native ads' (iklan terselubung) atau 'berita murni'?
A. native ads
B. berita murni

Jawaban (Pilih A atau B):
"""

GEMMA_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ '<start_of_turn>system\\n' + messages[0]['content'] + '<end_of_turn>\\n' }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<start_of_turn>user\\n' + message['content'] + '<end_of_turn>\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<start_of_turn>model\\n' + message['content'] + '<end_of_turn>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<start_of_turn>model\\n' }}"
    "{% endif %}"
)

def load_and_format_dataset(dataset_path: str, tokenizer) -> Dataset:
    # Try different path resolutions
    path = Path(dataset_path)
    if not path.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent
        path = script_dir / dataset_path
    
    if not path.exists():
        # Try relative to script's parent (project root)
        path = Path(__file__).parent.parent / "data" / Path(dataset_path).name

    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {dataset_path} or related locations.")

    print(f"Loading dataset from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    
    formatted_data = []
    for sample in data:
        # Simple extraction for title if not exists
        title = sample.get('title')
        if not title:
            title = sample.get('input', '')[:100]
        content = sample.get('input', '')[:400]
        
        gt_raw = sample.get('output', '').lower()
        if 'native ads' in gt_raw:
            answer = "A"
        else:
            answer = "B"
        
        user_text = TRAINING_PROMPT_TEMPLATE.format(title=title, content=content)
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted_data.append({"text": text})
        
    return Dataset.from_list(formatted_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../data/llm_dataset_12k_refined.json')
    parser.add_argument('--output_dir', type=str, default='../models/gemma-3-270m-hf-finetuned')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    model_id = "google/gemma-3-270m"
    hf_token = os.environ.get("HF_TOKEN")
    
    print(f"🚀 Loading {model_id} from Hugging Face...")
    
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check and set chat template if missing
    if not hasattr(tokenizer, 'chat_template') or not tokenizer.chat_template:
        print("Tokenizer is missing a chat_template. Setting the default Gemma chat template...")
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE
    
    # 2. Quantization config (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 3. Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token
    )
    model = prepare_model_for_kbit_training(model)
    
    # 4. LoRA config
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    
    # 5. Dataset
    dataset = load_and_format_dataset(args.dataset, tokenizer)
    
    # 6. Trainer
    from trl import SFTConfig
    
    config = SFTConfig(
        output_dir=args.output_dir,
        max_length=1024,
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        optim="paged_adamw_32bit",
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=config,
        processing_class=tokenizer,
    )
    
    print("⏳ Starting Training...")
    trainer.train()
    
    print(f"✅ Training complete. Saving to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
