# Quick Start - Fine-Tuning untuk A100 dengan MIG Mode

## 🚀 Setup Cepat (5 Menit)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Konversi Dataset
```bash
python tools/dataset_converter.py
# Tekan Enter 2x untuk default (content, label)
```

### 3. Fine-Tune Model (LoRA - Recommended)
```bash
python finetune_model.py \
  --model meta-llama/Llama-2-7b-hf \
  --dataset data/llm_dataset_instruction.json \
  --output models/native-ads-llama2-lora \
  --epochs 3 \
  --batch-size 4 \
  --use-lora
```

**Estimasi waktu**: 2-3 jam untuk 12,088 samples

### 4. Test Model
```bash
python run_local_demo.py \
  --model models/native-ads-llama2-lora \
  --url "https://www.cnnindonesia.com/ekonomi/20231212140523-532-1036329/promo-spesial-bri-di-hut-ke-128-diskon-hingga-rp1-28-juta"
```

## 🎯 Kenapa Ini Bekerja dengan MIG?

Script sudah diupdate untuk menggunakan:
- `device_map="auto"` - Auto handle MIG instances
- `torch_dtype=torch.float16` - Efficient memory usage
- **Tidak perlu sudo** untuk disable MIG

## 📊 Model Options

### Quick Test (Cepat)
```bash
--model microsoft/phi-2  # 2.7B params, ~15 menit
```

### Recommended (Balance)
```bash
--model meta-llama/Llama-2-7b-hf  # 7B params, ~2-3 jam
```

### Best Quality (Lambat)
```bash
--model meta-llama/Llama-2-13b-hf  # 13B params, ~5-6 jam
```

## 🔧 Jika Out of Memory

```bash
# Kurangi batch size
--batch-size 2

# Atau gunakan model lebih kecil
--model microsoft/phi-2
```

## 📈 Expected Results

Setelah fine-tuning:
- **Accuracy**: 85-95%
- **Precision**: 80-90%
- **Recall**: 80-90%

## 📚 Dokumentasi Lengkap

Lihat [`FINETUNING_GUIDE.md`](file:///Users/brianrizqi/Documents/Post%20Doc/agentic-ai-native-ads/FINETUNING_GUIDE.md) untuk detail lengkap.
