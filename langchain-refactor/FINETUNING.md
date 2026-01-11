# Fine-Tuning Module - Native Ads Detection

## Overview

Integrated fine-tuning module that uses **exact same prompts** as the LangChain inference system to ensure consistency.

## Files

- `finetune.py` - Main fine-tuning script
- `evaluate_model.py` - Model evaluation script
- `requirements-finetuning.txt` - Additional dependencies

## Quick Start

### 1. Prepare Dataset

```bash
# Convert dataset to correct JSON format
cd ..
python3 tools/fix_dataset_format.py \
  --input data/llm_dataset_instruction.json \
  --output data/llm_dataset_json_format.json
```

### 2. Fine-Tune Model

```bash
cd langchain-refactor

# With Qwen 2.5 14B (Recommended for Indonesian)
python finetune.py \
  --model "unsloth/Qwen2.5-14B-Instruct-bnb-4bit" \
  --dataset ../data/llm_dataset_json_format.json \
  --output ../models/native-ads-qwen14b \
  --max-steps 1500

# With Llama 3.1 8B
python finetune.py \
  --model "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" \
  --dataset ../data/llm_dataset_json_format.json \
  --output ../models/native-ads-llama8b \
  --max-steps 1500

# With GPT-OSS 20B (Experimental)
python finetune.py \
  --model "unsloth/gpt-oss-20b" \
  --dataset ../data/llm_dataset_json_format.json \
  --output ../models/native-ads-gptoss20b \
  --max-steps 1500 \
  --batch-size 1
```

### 3. Evaluate Model

```bash
python evaluate_model.py \
  --model ../models/native-ads-qwen14b_merged_16bit \
  --num-samples 100
```

### 4. Test with Real URL

```bash
python main.py \
  --url "https://tekno.sindonews.com/read/..." \
  --provider local \
  --model ../models/native-ads-qwen14b_merged_16bit
```

## Key Features

### ✅ Prompt Consistency
- Uses **exact same prompt** as `classification_prompts.py`
- No training/inference mismatch
- Model outputs correct JSON format

### ✅ Dataset Format
- Expects JSON format from `fix_dataset_format.py`
- Validates dataset balance
- Shows confidence distribution

### ✅ Unsloth Optimization
- 2x faster training
- 50% less memory
- QLoRA 4-bit quantization

### ✅ Evaluation Tools
- Accuracy metrics
- Per-class performance
- Confusion matrix
- Error analysis

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | Qwen2.5-14B | Base model |
| `--max-steps` | 1500 | Training steps |
| `--batch-size` | 1 | Per-device batch size |
| `--learning-rate` | 1e-4 | Learning rate |
| `--eval-steps` | 100 | Evaluation frequency |

## Expected Results

With 1500 steps on balanced dataset:
- **Accuracy**: 85-95%
- **Native Ads Recall**: 80-90%
- **Berita Murni Recall**: 85-95%
- **Training Time**: ~4 hours on A100 MIG

## Troubleshooting

### Model outputs wrong format
- Ensure dataset is from `fix_dataset_format.py`
- Check prompt consistency with `classification_prompts.py`

### Low accuracy
- Increase `--max-steps` to 2000-3000
- Check dataset balance
- Try different base model

### Out of memory
- Reduce `--batch-size` to 1
- Use smaller model (8B instead of 14B)
- Ensure 4-bit quantization is enabled

## Integration with Main System

After fine-tuning, the model can be used directly:

```python
from agents.classification_agent import ClassificationAgent

agent = ClassificationAgent(
    provider='local',
    model_name='../models/native-ads-qwen14b_merged_16bit'
)

result = agent.classify("Your article text here")
print(result)  # {'label': 'native ads', 'confidence': 0.87, 'reasoning': '...'}
```
