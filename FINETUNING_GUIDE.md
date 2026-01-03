# Fine-Tuning Guide - Native Ads Detection

## 🎯 Tujuan
Fine-tune model LLM dengan dataset native ads (12,088 samples) untuk meningkatkan akurasi deteksi.

## 📋 Prerequisites

### 1. Install Dependencies
```bash
pip install transformers datasets peft accelerate bitsandbytes
```

### 2. Prepare Dataset
```bash
# Konversi dataset Excel ke format LLM
python tools/dataset_converter.py

# Output: data/llm_dataset_instruction.json (12,088 samples)
```

## 🚀 Fine-Tuning Options

### Option 1: LoRA Fine-Tuning (RECOMMENDED - Efisien)

**Keuntungan**:
- Cepat (hanya train 1-2% parameter)
- Hemat VRAM
- Cocok untuk A100 dengan MIG mode

```bash
python finetune_model.py \
  --model meta-llama/Llama-2-7b-hf \
  --dataset data/llm_dataset_instruction.json \
  --output models/native-ads-llama2-lora \
  --epochs 3 \
  --batch-size 4 \
  --use-lora
```

**Estimasi waktu**: ~2-3 jam untuk 12,088 samples

### Option 2: Full Fine-Tuning

**Keuntungan**:
- Potensi akurasi lebih tinggi
- Full model adaptation

**Kekurangan**:
- Lebih lambat
- Butuh VRAM lebih besar

```bash
python finetune_model.py \
  --model meta-llama/Llama-2-7b-hf \
  --dataset data/llm_dataset_instruction.json \
  --output models/native-ads-llama2-full \
  --epochs 3 \
  --batch-size 2
```

## 🔧 Model Options

### Small Models (Cepat, Cocok untuk Testing)
```bash
# Phi-2 (2.7B parameters)
--model microsoft/phi-2

# Mistral 7B
--model mistralai/Mistral-7B-v0.1
```

### Medium Models (Balance)
```bash
# Llama 2 7B (RECOMMENDED)
--model meta-llama/Llama-2-7b-hf

# Llama 2 7B Chat
--model meta-llama/Llama-2-7b-chat-hf
```

### Large Models (Best Quality, Perlu VRAM Besar)
```bash
# Llama 2 13B
--model meta-llama/Llama-2-13b-hf
```

## 📊 Dataset Formats

Script mendukung 3 format dataset:

### 1. Instruction Format (RECOMMENDED)
```json
{
  "instruction": "Klasifikasikan artikel...",
  "input": "[konten]",
  "output": "**Klasifikasi**: native_ads..."
}
```

### 2. Chat Format
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### 3. QnA Format
```json
{
  "question": "Apakah artikel ini native ads?",
  "context": "[konten]",
  "answer": "Ya, ini native ads..."
}
```

## 🎛️ Training Parameters

### Untuk A100 40GB dengan MIG:
```bash
--batch-size 4          # Batch size per device
--epochs 3              # 3 epochs biasanya cukup
--use-lora              # Gunakan LoRA untuk efisiensi
```

### Jika Out of Memory:
```bash
--batch-size 2          # Kurangi batch size
--use-lora              # Pastikan LoRA enabled
```

## 📈 Monitoring Training

Training akan menampilkan:
- Loss per step
- Evaluation metrics setiap 100 steps
- Best model checkpoint

```
Step 100: loss=1.234, eval_loss=1.456
Step 200: loss=0.987, eval_loss=1.123
...
```

## ✅ Setelah Fine-Tuning

### 1. Test Model
```bash
python run_local_demo.py \
  --model models/native-ads-llama2-lora \
  --url "https://www.cnnindonesia.com/..."
```

### 2. Evaluate Performance
```bash
python tools/dataset_executor.py \
  --model models/native-ads-llama2-lora \
  --dataset data/llm_dataset_instruction.json \
  --num-samples 100
```

### 3. Compare dengan Baseline
```bash
# Test original model
python run_local_demo.py --model meta-llama/Llama-2-7b-hf --url "URL"

# Test fine-tuned model
python run_local_demo.py --model models/native-ads-llama2-lora --url "URL"
```

## 🔍 Troubleshooting

### Error: CUDA Out of Memory
```bash
# Solusi 1: Kurangi batch size
--batch-size 1

# Solusi 2: Gunakan LoRA
--use-lora

# Solusi 3: Gunakan model lebih kecil
--model microsoft/phi-2
```

### Error: MIG Mode Issue
Script sudah menggunakan `device_map="auto"` yang kompatibel dengan MIG. Tidak perlu disable MIG.

### Training Terlalu Lambat
```bash
# Gunakan LoRA
--use-lora

# Kurangi epochs
--epochs 1

# Test dengan subset data dulu
# Edit script untuk limit data[:1000]
```

## 📊 Expected Results

Dengan fine-tuning pada 12,088 samples:
- **Accuracy**: 85-95% (tergantung model)
- **Precision**: 80-90%
- **Recall**: 80-90%
- **F1-Score**: 80-90%

## 💡 Tips

1. **Start Small**: Test dulu dengan model kecil (phi-2) dan 1 epoch
2. **Use LoRA**: Lebih cepat dan efisien untuk A100 MIG
3. **Monitor Loss**: Jika loss tidak turun, coba adjust learning rate
4. **Save Checkpoints**: Training bisa di-resume jika terputus

## 🎯 Recommended Workflow

```bash
# 1. Konversi dataset
python tools/dataset_converter.py

# 2. Test dengan model kecil dulu (quick test)
python finetune_model.py \
  --model microsoft/phi-2 \
  --epochs 1 \
  --batch-size 4 \
  --use-lora

# 3. Jika berhasil, fine-tune model besar
python finetune_model.py \
  --model meta-llama/Llama-2-7b-hf \
  --epochs 3 \
  --batch-size 4 \
  --use-lora

# 4. Test hasil
python run_local_demo.py \
  --model models/native-ads-detector \
  --url "URL_TEST"
```
