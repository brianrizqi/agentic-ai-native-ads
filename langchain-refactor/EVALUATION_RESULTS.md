# Evaluation Results Organization

## 📁 New Folder Structure

All evaluation results are now saved in organized folders:

```
langchain-refactor/
└── eval_results/
    ├── eval_qwen_native_ads_v2_20260118_064500.json
    ├── eval_qwen_native_ads_v2_20260118_064500_summary.txt
    ├── eval_qwen_native_ads_v2_20260118_064500_confusion_matrix.png
    ├── eval_llama_native_ads_v2_20260118_070000.json
    ├── eval_llama_native_ads_v2_20260118_070000_summary.txt
    └── eval_llama_native_ads_v2_20260118_070000_confusion_matrix.png
```

## 🎯 Naming Convention

**Format**: `eval_{model_name}_{timestamp}`

- **model_name**: Extracted from model path (e.g., `qwen_native_ads_v2`)
- **timestamp**: `YYYYMMDD_HHMMSS` format

**Example**:
```
Model: ../models/qwen-native-ads-v2_merged_16bit
Output: eval_qwen_native_ads_v2_20260118_064500
```

## 📄 Output Files

For each evaluation, 3 files are created:

### 1. JSON Results (`*.json`)
Complete evaluation data including:
- Model metadata (name, path, timestamp)
- Overall metrics (accuracy, precision, recall)
- BERTScore results
- Per-class performance
- All predictions with reasoning

### 2. Summary Text (`*_summary.txt`)
Human-readable summary:
```
================================================================================
EVALUATION SUMMARY
================================================================================

Model: ../models/qwen-native-ads-v2_merged_16bit
Timestamp: 20260118_064500
Dataset: ../data/llm_dataset_finetuning_optimized.json
Samples: 200

Overall Metrics:
  Accuracy: 97.00%
  Correct: 194/200

BERTScore:
  Precision: 0.622
  Recall: 0.656
  F1: 0.638

Per-Class Accuracy:
  native ads: 97.00%
```

### 3. Confusion Matrix (`*_confusion_matrix.png`)
Visual confusion matrix plot

## 🚀 Usage

### Basic Evaluation
```bash
python3 evaluate_model.py \
  --model ../models/qwen-native-ads-v2_merged_16bit \
  --num-samples 200
```

**Output**:
```
eval_results/
├── eval_qwen_native_ads_v2_20260118_064500.json
├── eval_qwen_native_ads_v2_20260118_064500_summary.txt
└── eval_qwen_native_ads_v2_20260118_064500_confusion_matrix.png
```

### Custom Output Directory
```bash
python3 evaluate_model.py \
  --model ../models/llama-native-ads-v2_merged_16bit \
  --num-samples 200 \
  --output-dir ../my_evaluations
```

### Compare Multiple Models
```bash
# Evaluate Qwen
python3 evaluate_model.py \
  --model ../models/qwen-native-ads-v2_merged_16bit \
  --num-samples 200

# Evaluate Llama
python3 evaluate_model.py \
  --model ../models/llama-native-ads-v2_merged_16bit \
  --num-samples 200

# Evaluate Gemma
python3 evaluate_model.py \
  --model ../models/gemma-native-ads-v2_merged_16bit \
  --num-samples 200
```

All results saved in same folder for easy comparison!

## 📊 Benefits

1. **Organized**: All evaluations in one folder
2. **Timestamped**: Easy to track when evaluation was run
3. **Model-specific**: Clear which model was evaluated
4. **Multiple formats**: JSON (machine-readable), TXT (human-readable), PNG (visual)
5. **No overwriting**: Each run creates new files with unique timestamp

## 🔍 Finding Results

### Latest evaluation for a model:
```bash
ls -lt eval_results/eval_qwen_* | head -3
```

### All evaluations for a specific model:
```bash
ls eval_results/eval_qwen_*
```

### Compare accuracies:
```bash
grep "Accuracy:" eval_results/*_summary.txt
```
