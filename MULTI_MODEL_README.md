# Multi-Model Fine-Tuning Scripts

This directory contains scripts for fine-tuning and comparing 4 different models for native ads detection.

## 📚 Scripts

### `finetune_multi_model.py`
Unified fine-tuning script supporting 4 models:
- **Qwen 2.5 14B** - Best multilingual performance
- **GPT-OSS 20B** - Largest model
- **Llama 3.1 8B** - Balanced speed/quality
- **Gemma 2 9B** - Google's efficient model

### `compare_models.py`
Comprehensive model comparison with:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- Inference speed analysis
- Visualization plots

## 🚀 Usage

### Train Single Model

```bash
# Qwen 2.5 14B (recommended)
python finetune_multi_model.py --model qwen --max-steps 1500

# GPT-OSS 20B (largest)
python finetune_multi_model.py --model gpt-oss --max-steps 1500

# Llama 3.1 8B (fastest)
python finetune_multi_model.py --model llama --max-steps 1500

# Gemma 2 9B (efficient)
python finetune_multi_model.py --model gemma --max-steps 1500
```

### Train All Models

```bash
# Sequential training (~8-16 hours total)
bash scripts/train_all_models.sh
```

### Compare Models

```bash
cd langchain-refactor
python compare_models.py --num-samples 200
```

## 📊 Expected Outputs

### Models Directory
```
models/
├── qwen-native-ads/
│   └── merged_16bit/
├── gpt-oss-native-ads/
│   └── merged_16bit/
├── llama-native-ads/
│   └── merged_16bit/
└── gemma-native-ads/
    └── merged_16bit/
```

### Comparison Results
```
results/model_comparison/
├── comparison_results.json
├── model_comparison.csv
├── accuracy_comparison.png
├── f1_comparison.png
├── inference_speed_comparison.png
└── confusion_matrices.png
```

## 🎯 Model Comparison

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| Qwen 2.5 14B | 14B | Medium | Multilingual, best overall |
| GPT-OSS 20B | 20B | Slow | Highest quality |
| Llama 3.1 8B | 8B | Fast | Balanced performance |
| Gemma 2 9B | 9B | Fast | Efficient inference |

## 📝 For Research Paper

The comparison script generates publication-ready:
- **Tables:** CSV format for LaTeX
- **Plots:** High-res PNG (300 DPI)
- **Metrics:** JSON for further analysis

Perfect for the evaluation section of your paper!
