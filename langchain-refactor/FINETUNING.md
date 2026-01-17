# Fine-Tuning Documentation - Native Ads Classification

Dokumentasi lengkap untuk fine-tuning model klasifikasi native ads menggunakan Unsloth dan LoRA.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Preparation](#dataset-preparation)
4. [Model Selection](#model-selection)
5. [Training Process](#training-process)
6. [Evaluation](#evaluation)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## 🎯 Overview

Fine-tuning pipeline untuk melatih model LLM lokal (Qwen, Llama, Gemma) agar bisa mengklasifikasikan artikel sebagai **native ads** atau **berita murni** dengan akurasi tinggi.

**Hasil yang Dicapai:**
- ✅ Accuracy: **97%** (target: 85-90%)
- ✅ JSON Parse Success: **92.5%** (target: 95-98%)
- ✅ Reasoning: Bahasa Indonesia, concise (max 150 chars)

---

## 🔧 Prerequisites

### 1. Hardware Requirements

**Minimum:**
- GPU: NVIDIA dengan 24GB VRAM (e.g., RTX 3090, A5000)
- RAM: 32GB
- Storage: 50GB free space

**Recommended:**
- GPU: NVIDIA A100 (40GB/80GB) atau H100
- RAM: 64GB+
- Storage: 100GB+ SSD

### 2. Software Requirements

```bash
# Python 3.10+
python --version

# CUDA 11.8+ atau 12.1+
nvcc --version

# Install dependencies
pip install -r requirements-finetuning.txt
```

**Key Dependencies:**
- `unsloth` - Fast training framework
- `transformers` - HuggingFace models
- `trl` - Training utilities
- `peft` - LoRA adapters
- `datasets` - Dataset handling

---

## 📊 Dataset Preparation

### Step 1: Preprocess Dataset

**WAJIB dilakukan sebelum training!**

```bash
cd langchain-refactor

python3 tools/prepare_finetuning_dataset.py \
  --input ../data/llm_dataset_detailed.json \
  --output ../data/llm_dataset_finetuning_optimized.json \
  --max-reasoning 150
```

**Output:**
```
✅ 3000 samples processed
✅ Titles extracted: 3000
✅ Reasoning truncated: 1921 (64.0%)
   Original avg: 882 chars → New avg: 101 chars
✅ Dataset saved to: ../data/llm_dataset_finetuning_optimized.json
```

### Step 2: Validate Dataset

```bash
python3 -c "
import json
with open('../data/llm_dataset_finetuning_optimized.json') as f:
    data = json.load(f)
    
print(f'Total samples: {len(data)}')
print(f'Required fields: {list(data[0].keys())}')

# Check label distribution
labels = {}
for item in data:
    out = json.loads(item['output'])
    labels[out['label']] = labels.get(out['label'], 0) + 1
    
print(f'Label distribution:')
for label, count in labels.items():
    print(f'  {label}: {count} ({count/len(data)*100:.1f}%)')
"
```

**Expected Output:**
```
Total samples: 3000
Required fields: ['id', 'instruction', 'title', 'input', 'output']
Label distribution:
  native ads: 1500 (50.0%)
  berita murni: 1500 (50.0%)
```

---

## 🤖 Model Selection

### Available Models

| Model | Size | Speed | Quality | Bahasa Indonesia | Recommended For |
|-------|------|-------|---------|------------------|-----------------|
| **Qwen 2.5 14B** | 14B | Medium | ⭐⭐⭐⭐⭐ | Excellent | **Production** |
| **Llama 3.1 8B** | 8B | Fast | ⭐⭐⭐⭐ | Good | Development/Testing |
| **Gemma 2 9B** | 9B | Fast | ⭐⭐⭐⭐ | Good | Alternative |

### Model Configurations

Konfigurasi sudah dioptimasi di `finetune.py`:

```python
MODEL_CONFIGS = {
    "qwen": {
        "max_seq_length": 1024,      # Optimized for task
        "lora_r": 16,                 # LoRA rank
        "lora_alpha": 32,             # LoRA alpha
        "batch_size": 1,              # Per-device
        "gradient_accumulation": 4,   # Effective batch = 4
        "learning_rate": 2e-5,        # Stable training
    }
}
```

---

## 🚀 Training Process

### Quick Start

```bash
cd langchain-refactor

# Qwen 2.5 14B (Recommended)
python3 finetune.py \
  --model qwen \
  --dataset ../data/llm_dataset_finetuning_optimized.json \
  --output ../models/qwen-native-ads-v3 \
  --max-steps 1500
```

### Advanced Options

```bash
python3 finetune.py \
  --model qwen \
  --dataset ../data/llm_dataset_finetuning_optimized.json \
  --output ../models/qwen-native-ads-v3 \
  --max-steps 2000 \
  --epochs 1 \
  --eval-steps 100
```

**Parameters:**
- `--model`: Model type (`qwen`, `llama`, `gemma`)
- `--dataset`: Path to preprocessed dataset
- `--output`: Output directory for model
- `--max-steps`: Maximum training steps (default: 1500)
- `--epochs`: Number of epochs (default: 1)
- `--eval-steps`: Evaluation frequency (default: 100)

### Training Time

| Model | GPU | Steps | Time |
|-------|-----|-------|------|
| Qwen 14B | A100 40GB | 1500 | ~4-5 hours |
| Llama 8B | A100 40GB | 1500 | ~3-4 hours |
| Gemma 9B | A100 40GB | 1500 | ~3-4 hours |

### Output Files

Training menghasilkan 2 model variants:

```
models/
├── qwen-native-ads-v3/              # LoRA adapters (small)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
│
└── qwen-native-ads-v3_merged_16bit/ # Merged model (production-ready)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer files...
    └── training_curves.png
```

**Use merged model for inference!**

---

## 📊 Evaluation

### Basic Evaluation

```bash
cd langchain-refactor

python3 evaluate_model.py \
  --model ../models/qwen-native-ads-v3_merged_16bit \
  --num-samples 200
```

**Output:**
```
eval_results/
├── eval_qwen_native_ads_v3_20260118_065500.json
├── eval_qwen_native_ads_v3_20260118_065500_summary.txt
└── eval_qwen_native_ads_v3_20260118_065500_confusion_matrix.png
```

### Compare Multiple Models

```bash
# Train and evaluate all models
for model in qwen llama gemma; do
  python3 finetune.py --model $model --max-steps 1500
  python3 evaluate_model.py \
    --model ../models/${model}-native-ads-v3_merged_16bit \
    --num-samples 200
done

# Compare results
grep "Accuracy:" eval_results/*_summary.txt
```

---

## 🐛 Troubleshooting

### Issue 1: CUDA Out of Memory

**Solutions:**
```bash
# Option 1: Reduce batch size
python3 finetune.py --model qwen --batch-size 1

# Option 2: Use smaller model
python3 finetune.py --model llama  # 8B instead of 14B
```

### Issue 2: Low Accuracy (<80%)

**Solutions:**
```bash
# Increase training steps
python3 finetune.py --model qwen --max-steps 2500
```

### Issue 3: Model Generates English Reasoning

**Solution:** Prompt sudah diupdate (v3+), re-train model:
```bash
python3 finetune.py --model qwen --output ../models/qwen-native-ads-v3
```

---

## ✅ Best Practices

### 1. Dataset Quality

- ✅ **Balance classes**: 50-50 split
- ✅ **Truncate reasoning**: Max 150 characters
- ✅ **Extract titles**: Use preprocessing script
- ✅ **Validate format**: Check JSON structure

### 2. Training Configuration

- ✅ **Start with defaults**: Use recommended configs
- ✅ **Monitor eval loss**: Should decrease smoothly
- ✅ **Save checkpoints**: Every 200 steps
- ✅ **Use merged model**: For production

### 3. Evaluation

- ✅ **Test on unseen data**: Use last 200 samples
- ✅ **Check multiple metrics**: Accuracy, F1, JSON parse
- ✅ **Analyze errors**: Review misclassifications
- ✅ **Compare models**: Evaluate all variants

---

## 📚 Additional Resources

- [`FINETUNING_QUICKSTART.md`](FINETUNING_QUICKSTART.md) - Quick start
- [`EVALUATION_RESULTS.md`](EVALUATION_RESULTS.md) - Evaluation guide
- [`CHATBOT.md`](CHATBOT.md) - Using trained models

---

**Last Updated:** 2026-01-18  
**Version:** 3.0 (with Indonesian reasoning fix)
