# 🦙 Llama 3.2 Fine-Tuning Guide

## Mengapa Llama 3.2?

Qwen model mengalami bias ekstrem (prediksi semua sebagai native ads). Llama 3.2 lebih baik untuk:
- ✅ Better instruction following
- ✅ More balanced predictions
- ✅ Lebih stabil saat fine-tuning

## 🚀 Langkah-langkah Training

### 1. Copy Script ke Server

```bash
# Di komputer lokal
cd "/Users/brianrizqi/Documents/Post Doc/agentic-ai-native-ads"

scp finetune_llama_improved.py riset@58c85bc50301:/drive0-storage/agentic-ai-native-ads/
```

### 2. Jalankan Training di Server

```bash
# SSH ke server
ssh riset@58c85bc50301
cd /drive0-storage/agentic-ai-native-ads

# Start training (2-3 jam)
python finetune_llama_improved.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --dataset data/llm_dataset_instruction.json \
  --output models/native-ads-llama32-lora \
  --epochs 5 \
  --batch-size 4 \
  --learning-rate 5e-6
```

### 3. Monitor Progress

Training akan menampilkan:
- Loss per step
- Evaluation metrics setiap 100 steps
- Estimated time remaining

**Expected training time**: 2-3 hours on A100 MIG

### 4. Evaluate Hasil

Setelah training selesai:

```bash
# Evaluate model baru
python comprehensive_evaluation_full.py \
  --model models/native-ads-llama32-lora \
  --dataset data/llm_dataset_instruction.json \
  --num-samples 500 \
  --output data/eval_llama32.json
```

### 5. Compare dengan Qwen

```bash
# Compare results
python -c "
import json

print('=== QWEN (Old) ===')
with open('data/eval_qwen_full.json') as f:
    qwen = json.load(f)
    print(f\"Accuracy: {qwen['metrics']['traditional']['accuracy']*100:.1f}%\")
    print(f\"Berita Murni Recall: {qwen['metrics']['traditional']['berita_murni']['recall']*100:.1f}%\")

print('\n=== LLAMA 3.2 (New) ===')
with open('data/eval_llama32.json') as f:
    llama = json.load(f)
    print(f\"Accuracy: {llama['metrics']['traditional']['accuracy']*100:.1f}%\")
    print(f\"Berita Murni Recall: {llama['metrics']['traditional']['berita_murni']['recall']*100:.1f}%\")
"
```

## 🔧 Improvements dalam Script

### 1. **Lower Learning Rate**
- Qwen: `2e-5` (terlalu tinggi → overfit)
- Llama: `5e-6` (lebih stabil)

### 2. **More Epochs**
- Qwen: `3` epochs
- Llama: `5` epochs (lebih banyak iterasi)

### 3. **Label Smoothing**
- Qwen: None
- Llama: `0.1` (cegah overconfidence)

### 4. **Gradient Clipping**
- Qwen: None
- Llama: `max_grad_norm=1.0` (stabilize training)

### 5. **Better LR Schedule**
- Qwen: Linear
- Llama: Cosine (smoother decay)

### 6. **Improved Prompt in Training Data**
- Tambah instruksi: "TIDAK semua konten adalah native ads"
- Balanced examples dalam format training

## 📊 Expected Results

| Metric | Qwen | Llama (Expected) |
|--------|------|------------------|
| **Accuracy** | 47.6% | **75-85%** |
| **Native Ads Recall** | 98.7% | **75-85%** |
| **Berita Murni Recall** | 0.77% | **70-80%** |
| **Macro F1** | 0.16 | **0.70+** |

## ⚠️ Troubleshooting

### Jika Training Error

**Error: CUDA out of memory**
```bash
# Reduce batch size
python finetune_llama_improved.py --batch-size 2
```

**Error: Model not found**
```bash
# Login to HuggingFace first
huggingface-cli login
# Paste your token
```

**Error: MIG mode issue**
```bash
# Script sudah handle MIG dengan device_map="auto"
# Tapi kalau masih error, set:
export CUDA_VISIBLE_DEVICES=0
```

## 🎯 Next Steps After Training

1. **Evaluate** model baru
2. **Compare** dengan Qwen
3. **Test** dengan real URLs
4. **Deploy** jika hasilnya bagus (>75% accuracy)

## 📝 Notes

- Training menggunakan **LoRA** (efficient fine-tuning)
- Model size: ~3B parameters
- Trainable params: ~16M (hanya 0.5% dari total)
- Memory usage: ~12GB GPU RAM
