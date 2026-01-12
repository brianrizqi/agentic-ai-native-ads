# Quick Start Guide - Multi-Model Fine-Tuning

## 🚀 One-Command Training

Train all 4 models sequentially with a single command:

```bash
cd langchain-refactor
./train_all_models.sh
```

This will train:
1. **Llama 3.1 8B** (~2-3 hours)
2. **Gemma 2 9B** (~2-3 hours)
3. **Qwen 2.5 14B** (~3-4 hours)
4. **GPT-OSS 20B** (~4-5 hours)

**Total time:** ~8-16 hours

## 📋 What It Does

The script will:
- ✅ Check dataset exists
- ✅ Train each model sequentially
- ✅ Log all output to file
- ✅ Track training times
- ✅ Save all models

## 📊 Output

### Models
```
models/
├── llama-native-ads_merged_16bit/
├── gemma-native-ads_merged_16bit/
├── qwen-native-ads_merged_16bit/
└── gpt-oss-native-ads_merged_16bit/
```

### Logs
```
results/multi_model_training/
├── training_log_YYYYMMDD_HHMMSS.txt
└── training_times.txt
```

## 🎯 After Training

### 1. Compare Models
```bash
cd langchain-refactor
python compare_models.py --num-samples 200
```

### 2. Test a Model
```bash
cd langchain-refactor
python main.py \
  --url "https://tekno.sindonews.com/..." \
  --provider local \
  --model ../models/qwen-native-ads_merged_16bit
```

### 3. View Results
```bash
cat results/model_comparison/model_comparison.csv
```

## 🛑 Stop Training

Press `Ctrl+C` to stop the current model training.

The script will exit and you can resume later by running individual models:

```bash
cd langchain-refactor
python finetune.py --model qwen --max-steps 1500
```

## 💡 Tips

- **Run overnight:** Training takes 8-16 hours
- **Check GPU:** Make sure CUDA is available
- **Monitor:** Use `nvidia-smi` to check GPU usage
- **Logs:** Check `results/multi_model_training/` for progress

## 🔧 Troubleshooting

### Dataset not found
```bash
# Prepare dataset first
python tools/merge_datasets.py \
  --inputs data/llm_dataset_instruction.json data/llm_dataset_detailed.json \
  --output data/llm_dataset_mixed.json

python tools/fix_dataset_format.py \
  --input data/llm_dataset_mixed.json \
  --output data/llm_dataset_mixed_json.json
```

### Out of memory
Edit `langchain-refactor/finetune.py` and reduce batch size in MODEL_CONFIGS

### Training too slow
Reduce `--max-steps` to 1000 or 500 for faster training

## 📈 Expected Results

| Model | Accuracy | Speed | Memory |
|-------|----------|-------|--------|
| Llama 3.1 8B | ~85% | Fast | 16GB |
| Gemma 2 9B | ~85% | Fast | 18GB |
| Qwen 2.5 14B | ~88% | Medium | 24GB |
| GPT-OSS 20B | ~90% | Slow | 32GB |

Perfect for research paper comparison! 🎓
