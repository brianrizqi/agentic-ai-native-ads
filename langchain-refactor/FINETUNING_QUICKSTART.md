# Fine-Tuning Quick Start Guide (UPDATED)

## 🚀 Langkah-Langkah Fine-Tuning yang Sudah Diperbaiki

### 1. Preprocessing Dataset (WAJIB!)

```bash
cd langchain-refactor

# Preprocess dataset untuk extract title dan truncate reasoning
python3 tools/prepare_finetuning_dataset.py \
  --input ../data/llm_dataset_detailed.json \
  --output ../data/llm_dataset_finetuning_optimized.json \
  --max-reasoning 150
```

**Output yang diharapkan:**
- ✅ 3000 samples processed
- ✅ Titles extracted: 3000
- ✅ Reasoning truncated: ~64% (dari avg 882 chars → 101 chars)
- ✅ Dataset saved to: `../data/llm_dataset_finetuning_optimized.json`

### 2. Fine-Tune Model

```bash
cd langchain-refactor

# Fine-tune Qwen 2.5 14B (Recommended)
python3 finetune.py \
  --model qwen \
  --dataset ../data/llm_dataset_finetuning_optimized.json \
  --output ../models/qwen-native-ads-v2 \
  --max-steps 1500 \
  --epochs 1

# Atau Llama 3.1 8B (Lebih cepat)
python3 finetune.py \
  --model llama \
  --dataset ../data/llm_dataset_finetuning_optimized.json \
  --output ../models/llama-native-ads-v2 \
  --max-steps 1500

# Atau Gemma 2 9B
python3 finetune.py \
  --model gemma \
  --dataset ../data/llm_dataset_finetuning_optimized.json \
  --output ../models/gemma-native-ads-v2 \
  --max-steps 1500
```

**Training akan:**
- Use optimized config (1024 seq length, 2e-5 LR)
- Train/eval split: 90/10
- Save checkpoints setiap 200 steps
- Generate training curves

### 3. Evaluate Model

```bash
# Evaluate dengan test set
python3 evaluate_model.py \
  --model ../models/qwen-native-ads-v2_merged_16bit \
  --num-samples 200

# Expected results (IMPROVED):
# - Accuracy: ≥ 85% (was ~70-75%)
# - F1-Score: ≥ 0.83 (was ~0.68)
# - JSON Parse Success: ≥ 95% (was ~60-70%)
```

### 4. Test dengan Real URL

```bash
# Test inference
python3 main.py \
  --url "https://tekno.sindonews.com/read/..." \
  --provider local \
  --model ../models/qwen-native-ads-v2_merged_16bit
```

### 5. Compare Multiple Models (Optional)

```bash
# Update compare_models.py dengan path model baru
python3 compare_models.py \
  --dataset ../data/llm_dataset_finetuning_optimized.json \
  --num-samples 200 \
  --output ../results/model_comparison_v2
```

---

## 🔧 Apa yang Sudah Diperbaiki?

### ✅ Fix 1: Prompt Consistency
- **Before**: Training tidak pakai `title`, inference pakai `title` ❌
- **After**: Training dan inference sama-sama pakai `title` ✅

### ✅ Fix 2: Reasoning Length
- **Before**: Avg 882 chars (max 1840 chars) - terlalu panjang ❌
- **After**: Avg 101 chars (max 152 chars) - optimal ✅

### ✅ Fix 3: Training Config
- **Before**: 2048 seq length, 1e-4 LR, LoRA r=8 ❌
- **After**: 1024 seq length, 2e-5 LR, LoRA r=16 ✅

### ✅ Fix 4: Dataset Format
- **Before**: Tidak ada title field ❌
- **After**: Title extracted dari input ✅

---

## 📊 Expected Improvements

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Accuracy | 70-75% | **85-90%** |
| F1-Score | 0.68 | **0.83-0.88** |
| JSON Parse | 60-70% | **95-98%** |
| Reasoning | Inconsistent | Concise & consistent |

---

## 🐛 Troubleshooting

### Error: "Dataset not found"
```bash
# Run preprocessing first
python3 tools/prepare_finetuning_dataset.py
```

### Error: "CUDA out of memory"
```bash
# Reduce batch size in finetune.py
# Or use smaller model (llama instead of qwen)
```

### Low accuracy after training
- Check training curves (`training_curves.png`)
- Increase max-steps to 2000-3000
- Try different model (qwen vs llama vs gemma)

---

## 📝 Notes

- **Training time**: ~4-6 hours on A100 MIG (1500 steps)
- **Disk space**: ~30GB per model (merged 16bit)
- **Memory**: ~24GB VRAM untuk Qwen 14B, ~16GB untuk Llama 8B

---

## 🎯 Next Steps After Fine-Tuning

1. **Evaluate thoroughly**
   ```bash
   python3 evaluate_model.py --model ../models/qwen-native-ads-v2_merged_16bit --num-samples 200
   ```

2. **Test with chatbot**
   ```bash
   python3 chatbot.py --provider local --model ../models/qwen-native-ads-v2_merged_16bit
   ```

3. **Compare with baseline**
   ```bash
   python3 compare_models.py
   ```

4. **Deploy to production** (if results good)
   - Copy model to production server
   - Update API endpoints
   - Monitor performance
