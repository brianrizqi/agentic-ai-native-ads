# Dataset Conversion Guide

This tool converts your raw Excel dataset (`native_ads_dataset.xlsx`) into various JSON formats suitable for:
1.  **RAG (Retrieval-Augmented Generation)**: Using `llm_dataset_qna.json`.
2.  **Fine-tuning**: Using `llm_dataset_instruction.jsonl` (for Llama/Mistral/GPT-OSS).
3.  **Chat Fine-tuning**: Using `llm_dataset_chat.jsonl` (for OpenAI format).

## Usage

### 1. Preparation
Ensure dependencies are installed:
```bash
pip install pandas openpyxl transformers torch
```

### 2. Running Converter
```bash
python tools/dataset_converter.py
```

### 3. Output Formats

| File | Format | Usage |
|------|--------|-------|
| `llm_dataset_qna.json` | JSON | Knowledge Base for RAG (Retriever Agent). Contains Question-Answer pairs. |
| `llm_dataset_instruction.jsonl` | JSONL | Fine-tuning dataset for Open Source models (HuggingFace). |

## Fine-tuning Tips
To fine-tune `openai/gpt-oss-20b` or other open models:
1. Use the `llm_dataset_instruction.jsonl` file.
2. Use HuggingFace TRL (Transformer Reinforcement Learning) library or similar.
3. Example library: [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) or [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory).

## Data Augmentation
Use `tools/llm_augmentation.py` to generate more variations of questions/explanations using HuggingFace models.
