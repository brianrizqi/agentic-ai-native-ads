# Evaluation Guide - BERTScore & LLM-as-a-Judge

## 📊 Overview

Script evaluasi komprehensif untuk mengukur performa sistem Agentic AI menggunakan:

1. **BERTScore** - Semantic similarity antara reasoning yang dihasilkan dengan ground truth
2. **LLM-as-a-Judge** - Evaluasi kualitas klasifikasi oleh LLM lain
3. **Traditional Metrics** - Accuracy, Precision, Recall, F1-Score

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install bert-score scikit-learn tqdm
```

### 2. Run Evaluation
```bash
# Evaluate baseline model (100 samples)
python evaluate_model.py --num-samples 100

# Evaluate fine-tuned model
python evaluate_model.py \
  --model models/native-ads-llama2-lora \
  --num-samples 100

# Full evaluation (all 12,088 samples - will take hours)
python evaluate_model.py --num-samples 12088
```

## 📈 Evaluation Metrics

### 1. Traditional Metrics
- **Accuracy**: Percentage of correct classifications
- **Average Confidence**: Mean confidence score across predictions

### 2. BERTScore
Mengukur semantic similarity antara reasoning yang dihasilkan dengan ground truth:
- **Precision**: Seberapa relevan reasoning yang dihasilkan
- **Recall**: Seberapa lengkap reasoning mencakup ground truth
- **F1**: Harmonic mean dari precision dan recall

**Interpretasi**:
- Score > 0.9: Excellent
- Score 0.8-0.9: Good
- Score 0.7-0.8: Fair
- Score < 0.7: Needs improvement

### 3. LLM-as-a-Judge Scores (1-5 scale)

Model judge mengevaluasi 4 aspek:

1. **Correctness Score**: Apakah label prediksi benar?
   - 5: Completely correct
   - 3: Partially correct
   - 1: Completely wrong

2. **Confidence Calibration Score**: Apakah confidence level sesuai?
   - 5: Perfectly calibrated
   - 3: Somewhat calibrated
   - 1: Poorly calibrated

3. **Reasoning Quality Score**: Apakah reasoning jelas dan logis?
   - 5: Excellent reasoning
   - 3: Acceptable reasoning
   - 1: Poor reasoning

4. **Overall Quality Score**: Penilaian keseluruhan
   - 5: Excellent
   - 3: Acceptable
   - 1: Poor

## 📊 Example Output

```
================================================================================
EVALUATION SUMMARY
================================================================================

Model: mistralai/Mistral-7B-Instruct-v0.2
Samples Evaluated: 100

--- Traditional Metrics ---
Accuracy: 87.00%
Average Confidence: 0.82

--- BERTScore ---
Precision: 0.8543
Recall: 0.8621
F1: 0.8582

--- LLM-as-a-Judge Scores (1-5 scale) ---
Correctness: 4.23/5.0
Confidence Calibration: 3.87/5.0
Reasoning Quality: 4.01/5.0
Overall Quality: 4.12/5.0

================================================================================
```

## 🔬 For Research

### Baseline vs Fine-Tuned Comparison

```bash
# 1. Evaluate baseline
python evaluate_model.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --num-samples 500 \
  --output data/eval_baseline.json

# 2. Evaluate fine-tuned
python evaluate_model.py \
  --model models/native-ads-llama2-lora \
  --num-samples 500 \
  --output data/eval_finetuned.json

# 3. Compare results
python -c "
import json
with open('data/eval_baseline.json') as f:
    baseline = json.load(f)
with open('data/eval_finetuned.json') as f:
    finetuned = json.load(f)

print('Baseline Accuracy:', baseline['traditional_metrics']['accuracy'])
print('Fine-tuned Accuracy:', finetuned['traditional_metrics']['accuracy'])
print('Improvement:', finetuned['traditional_metrics']['accuracy'] - baseline['traditional_metrics']['accuracy'])
"
```

### Different Judge Models

```bash
# Use different model as judge for unbiased evaluation
python evaluate_model.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --judge-model meta-llama/Llama-2-7b-chat-hf \
  --num-samples 100
```

## 📝 Output Files

Evaluation results disimpan dalam JSON dengan struktur:

```json
{
  "model": "model_name",
  "num_samples": 100,
  "traditional_metrics": {
    "accuracy": 0.87,
    "avg_confidence": 0.82
  },
  "bertscore": {
    "precision": 0.8543,
    "recall": 0.8621,
    "f1": 0.8582
  },
  "llm_judge_scores": {
    "correctness": 4.23,
    "confidence_calibration": 3.87,
    "reasoning_quality": 4.01,
    "overall_quality": 4.12
  },
  "detailed_results": [...],
  "judge_evaluations": [...]
}
```

## 🎯 Expected Results

### Baseline Model (Pre-trained)
- Accuracy: 70-80%
- BERTScore F1: 0.75-0.85
- LLM Judge Overall: 3.5-4.0/5.0

### Fine-tuned Model
- Accuracy: 85-95%
- BERTScore F1: 0.85-0.92
- LLM Judge Overall: 4.0-4.5/5.0

## ⚙️ Advanced Options

### Custom Dataset
```bash
python evaluate_model.py \
  --dataset path/to/custom_dataset.json \
  --num-samples 200
```

### Specific Evaluation
```bash
# Only evaluate on native ads samples
# (requires filtering dataset first)
python evaluate_model.py \
  --dataset data/native_ads_only.json
```

## 🔍 Troubleshooting

### BERTScore Slow
BERTScore menggunakan BERT model untuk scoring, bisa lambat untuk dataset besar:
```bash
# Reduce samples
--num-samples 50

# Or disable BERTScore (edit script)
```

### LLM Judge Timeout
Jika judge model terlalu lambat:
```bash
# Use smaller judge model
--judge-model microsoft/phi-2
```

## 📊 For Publication

Untuk paper penelitian, jalankan evaluasi lengkap:

```bash
# Full evaluation with multiple metrics
python evaluate_model.py \
  --model models/native-ads-final \
  --judge-model mistralai/Mistral-7B-Instruct-v0.2 \
  --num-samples 1000 \
  --output data/final_evaluation.json
```

Hasil bisa digunakan untuk:
- Tabel perbandingan performa
- Grafik accuracy vs confidence
- Analisis error cases
- Statistical significance testing
