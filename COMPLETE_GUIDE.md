# Complete Guide - Agentic AI Native Ads Detection
## Panduan Lengkap dari Setup hingga Evaluasi

**Penelitian Postdoc**: Pengembangan Agentic AI untuk Deteksi Native Ads pada Portal Berita Elektronik

---

## 📋 Table of Contents

1. [Setup Environment](#1-setup-environment)
2. [Konversi Dataset](#2-konversi-dataset)
3. [Testing Sistem](#3-testing-sistem)
4. [Fine-Tuning Model](#4-fine-tuning-model)
5. [Evaluasi Model](#5-evaluasi-model)
6. [Analisis Hasil](#6-analisis-hasil)

---

## 1. Setup Environment

### 1.1 Install Python Dependencies

```bash
# Pastikan di direktori project
cd /path/to/agentic-ai-native-ads

# Install semua dependencies
pip install -r requirements.txt
```

**Dependencies yang diinstall**:
- `transformers` - HuggingFace models
- `torch` - PyTorch untuk deep learning
- `sentence-transformers` - Embeddings untuk RAG
- `peft` - LoRA fine-tuning
- `bert-score` - Evaluasi semantic similarity
- Dan lainnya (lihat `requirements.txt`)

### 1.2 Verifikasi GPU (Opsional)

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
```

**Catatan**: Sistem bisa jalan di CPU, tapi akan lebih lambat.

---

## 2. Konversi Dataset

### 2.1 Prepare Dataset

Pastikan file `native_ads_dataset.xlsx` ada di root directory dengan:
- Sheet name: `Clean`
- Columns: `content`, `label`
- Total: 12,088 samples

### 2.2 Run Converter

```bash
python tools/dataset_converter.py
```

**Interaksi**:
```
Nama kolom untuk konten artikel (default: 'content'): [Enter]
Nama kolom untuk label (default: 'label'): [Enter]
```

**Output Files**:
- `data/llm_dataset_qna.json` - Untuk RAG (Retriever Agent)
- `data/llm_dataset_instruction.json` - Untuk fine-tuning
- `data/llm_dataset_instruction.jsonl` - Format JSONL
- `data/llm_dataset_chat.jsonl` - Format chat (OpenAI)

### 2.3 Verify Conversion

```bash
# Check file size
ls -lh data/

# Check sample data
python -c "
import json
with open('data/llm_dataset_qna.json') as f:
    data = json.load(f)
print(f'Total samples: {len(data)}')
print(f'Sample: {data[0]}')
"
```

---

## 3. Testing Sistem

### 3.1 Test Single URL

```bash
python run_local_demo.py --url "https://www.cnnindonesia.com/teknologi/20221203112947-190-882335/jadwalkan-rapat-kian-mudah-cek-cara-bagikan-google-calendar"
```

**Proses yang terjadi**:
1. **Web Agent** - Scraping artikel dari URL
2. **Preprocessing Agent** - Cleaning text, extract features
3. **Retriever Agent** - Cari 5 dokumen relevan dari knowledge base
4. **LLM Classifier** - Klasifikasi (Native Ads/Editorial)
5. **Explanation Agent** - Generate penjelasan

**Output**:
```
--- CLASSIFICATION RESULT ---
Label: Native Advertising
Confidence: 92.00%
Reasoning: Artikel mengandung promosi produk BRI...

--- EXPLANATION ---
[Penjelasan detail...]

[OK] Results saved to local_demo_result.json
```

### 3.2 Check Results

```bash
# View hasil
cat local_demo_result.json | python -m json.tool
```

### 3.3 Test Multiple URLs

Buat file `test_urls.txt`:
```
https://www.cnnindonesia.com/ekonomi/...
https://www.kompas.com/...
https://www.detik.com/...
```

Lalu jalankan:
```bash
while read url; do
  python run_local_demo.py --url "$url"
  sleep 2
done < test_urls.txt
```

---

## 4. Fine-Tuning Model

### 4.1 Quick Test (Recommended First)

Test dengan model kecil dan 1 epoch:

```bash
python finetune_model.py \
  --model microsoft/phi-2 \
  --epochs 1 \
  --batch-size 4 \
  --use-lora \
  --output models/test-finetuned
```

**Estimasi waktu**: ~15-30 menit

### 4.2 Full Fine-Tuning (Production)

```bash
python finetune_model.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --dataset data/llm_dataset_instruction.json \
  --output models/native-ads-mistral-lora \
  --epochs 3 \
  --batch-size 4 \
  --use-lora
```

**Estimasi waktu**: 2-4 jam (tergantung GPU)

**Parameter Explanation**:
- `--model`: Base model untuk fine-tune
- `--dataset`: Dataset yang sudah dikonversi
- `--output`: Direktori output model
- `--epochs`: Jumlah epoch training (3 recommended)
- `--batch-size`: Batch size (4 untuk GPU 40GB)
- `--use-lora`: Gunakan LoRA (efisien, recommended)

### 4.3 Monitor Training

Training akan menampilkan:
```
[1/6] Loading tokenizer...
[2/6] Loading model (using device_map='auto' for MIG compatibility)...
[3/6] Applying LoRA configuration...
trainable params: 4,194,304 || all params: 7,241,732,096 || trainable%: 0.0579
[4/6] Loading and preparing dataset...
Training samples: 10879
Evaluation samples: 1209
[5/6] Setting up training configuration...
[6/6] Starting training...

Step 100: loss=1.234, eval_loss=1.456
Step 200: loss=0.987, eval_loss=1.123
...
```

### 4.4 Test Fine-Tuned Model

```bash
python run_local_demo.py \
  --model models/native-ads-mistral-lora \
  --url "https://www.cnnindonesia.com/..."
```

---

## 5. Evaluasi Model

### 5.1 Install Evaluation Dependencies

```bash
pip install bert-score scikit-learn tqdm
```

### 5.2 Evaluate Baseline Model

```bash
python evaluate_model.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --num-samples 100 \
  --output data/eval_baseline.json
```

**Metrics yang dihitung**:
1. **Traditional Metrics**: Accuracy, Average Confidence
2. **BERTScore**: Precision, Recall, F1 (semantic similarity)
3. **LLM-as-a-Judge**: 4 scores (correctness, confidence calibration, reasoning quality, overall)

### 5.3 Evaluate Fine-Tuned Model

```bash
python evaluate_model.py \
  --model models/native-ads-mistral-lora \
  --num-samples 100 \
  --output data/eval_finetuned.json
```

### 5.4 Full Evaluation (For Publication)

```bash
# Evaluate dengan 1000 samples untuk hasil yang robust
python evaluate_model.py \
  --model models/native-ads-mistral-lora \
  --num-samples 1000 \
  --output data/eval_final.json
```

**Estimasi waktu**: 2-4 jam untuk 1000 samples

---

## 6. Analisis Hasil

### 6.1 View Evaluation Results

```bash
# Pretty print JSON
cat data/eval_finetuned.json | python -m json.tool | less
```

### 6.2 Compare Baseline vs Fine-Tuned

```bash
python -c "
import json

# Load results
with open('data/eval_baseline.json') as f:
    baseline = json.load(f)
with open('data/eval_finetuned.json') as f:
    finetuned = json.load(f)

# Compare
print('='*60)
print('COMPARISON: Baseline vs Fine-Tuned')
print('='*60)
print()
print('Traditional Metrics:')
print(f'  Baseline Accuracy:    {baseline[\"traditional_metrics\"][\"accuracy\"]:.2%}')
print(f'  Fine-tuned Accuracy:  {finetuned[\"traditional_metrics\"][\"accuracy\"]:.2%}')
print(f'  Improvement:          {(finetuned[\"traditional_metrics\"][\"accuracy\"] - baseline[\"traditional_metrics\"][\"accuracy\"]):.2%}')
print()
print('BERTScore F1:')
print(f'  Baseline:    {baseline[\"bertscore\"][\"f1\"]:.4f}')
print(f'  Fine-tuned:  {finetuned[\"bertscore\"][\"f1\"]:.4f}')
print(f'  Improvement: {(finetuned[\"bertscore\"][\"f1\"] - baseline[\"bertscore\"][\"f1\"]):.4f}')
print()
print('LLM Judge Overall Quality (1-5):')
print(f'  Baseline:    {baseline[\"llm_judge_scores\"][\"overall_quality\"]:.2f}/5.0')
print(f'  Fine-tuned:  {finetuned[\"llm_judge_scores\"][\"overall_quality\"]:.2f}/5.0')
print()
"
```

### 6.3 Export Results for Paper

```bash
# Create summary table
python -c "
import json
import pandas as pd

with open('data/eval_baseline.json') as f:
    baseline = json.load(f)
with open('data/eval_finetuned.json') as f:
    finetuned = json.load(f)

# Create comparison table
data = {
    'Metric': [
        'Accuracy',
        'Avg Confidence',
        'BERTScore Precision',
        'BERTScore Recall',
        'BERTScore F1',
        'LLM Judge - Correctness',
        'LLM Judge - Confidence Calibration',
        'LLM Judge - Reasoning Quality',
        'LLM Judge - Overall Quality'
    ],
    'Baseline': [
        f\"{baseline['traditional_metrics']['accuracy']:.2%}\",
        f\"{baseline['traditional_metrics']['avg_confidence']:.2%}\",
        f\"{baseline['bertscore']['precision']:.4f}\",
        f\"{baseline['bertscore']['recall']:.4f}\",
        f\"{baseline['bertscore']['f1']:.4f}\",
        f\"{baseline['llm_judge_scores']['correctness']:.2f}/5.0\",
        f\"{baseline['llm_judge_scores']['confidence_calibration']:.2f}/5.0\",
        f\"{baseline['llm_judge_scores']['reasoning_quality']:.2f}/5.0\",
        f\"{baseline['llm_judge_scores']['overall_quality']:.2f}/5.0\"
    ],
    'Fine-Tuned': [
        f\"{finetuned['traditional_metrics']['accuracy']:.2%}\",
        f\"{finetuned['traditional_metrics']['avg_confidence']:.2%}\",
        f\"{finetuned['bertscore']['precision']:.4f}\",
        f\"{finetuned['bertscore']['recall']:.4f}\",
        f\"{finetuned['bertscore']['f1']:.4f}\",
        f\"{finetuned['llm_judge_scores']['correctness']:.2f}/5.0\",
        f\"{finetuned['llm_judge_scores']['confidence_calibration']:.2f}/5.0\",
        f\"{finetuned['llm_judge_scores']['reasoning_quality']:.2f}/5.0\",
        f\"{finetuned['llm_judge_scores']['overall_quality']:.2f}/5.0\"
    ]
}

df = pd.DataFrame(data)
print(df.to_markdown(index=False))
" > data/comparison_table.md

cat data/comparison_table.md
```

---

## 📊 Expected Results

### Baseline Model (Pre-trained Mistral-7B)
- **Accuracy**: 70-80%
- **BERTScore F1**: 0.75-0.85
- **LLM Judge Overall**: 3.5-4.0/5.0

### Fine-Tuned Model (After 3 epochs)
- **Accuracy**: 85-95%
- **BERTScore F1**: 0.85-0.92
- **LLM Judge Overall**: 4.0-4.5/5.0

### Expected Improvement
- **Accuracy**: +10-15%
- **BERTScore F1**: +0.05-0.10
- **LLM Judge**: +0.5-1.0 points

---

## 🔧 Troubleshooting

### Issue 1: CUDA Out of Memory
```bash
# Solusi: Kurangi batch size
python finetune_model.py --batch-size 2 --use-lora
```

### Issue 2: Model Download Slow
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/large/disk
```

### Issue 3: Evaluation Too Slow
```bash
# Reduce samples
python evaluate_model.py --num-samples 50
```

---

## 📚 File Structure

```
agentic-ai-native-ads/
├── agents/                          # 6 AI Agents
│   ├── web_agent.py                # Web scraping
│   ├── preprocessing_agent.py      # Text cleaning
│   ├── retriever_agent.py          # RAG
│   ├── llm_classifier_agent.py     # Classification
│   ├── explanation_agent.py        # Explanation
│   └── feedback_retrainer_agent.py # Feedback
├── tools/                           # Utilities
│   └── dataset_converter.py        # Dataset conversion
├── data/                            # Data files
│   ├── llm_dataset_qna.json       # RAG dataset
│   ├── llm_dataset_instruction.json # Training dataset
│   ├── eval_baseline.json          # Baseline results
│   └── eval_finetuned.json         # Fine-tuned results
├── models/                          # Fine-tuned models
│   └── native-ads-mistral-lora/   # Your fine-tuned model
├── run_local_demo.py               # Main demo script
├── finetune_model.py               # Fine-tuning script
├── evaluate_model.py               # Evaluation script
├── config.json                      # Configuration
├── requirements.txt                 # Dependencies
└── COMPLETE_GUIDE.md               # This file
```

---

## 🎓 For Research Publication

### Recommended Workflow

1. **Baseline Evaluation** (100-500 samples)
2. **Fine-Tuning** (3 epochs dengan LoRA)
3. **Full Evaluation** (1000+ samples)
4. **Statistical Analysis** (t-test, confidence intervals)
5. **Error Analysis** (manual review of misclassifications)

### Metrics to Report

**Table 1: Classification Performance**
| Metric | Baseline | Fine-Tuned | Improvement |
|--------|----------|------------|-------------|
| Accuracy | XX% | XX% | +XX% |
| Precision | XX% | XX% | +XX% |
| Recall | XX% | XX% | +XX% |
| F1-Score | XX% | XX% | +XX% |

**Table 2: Semantic Quality (BERTScore)**
| Metric | Baseline | Fine-Tuned |
|--------|----------|------------|
| Precision | X.XXX | X.XXX |
| Recall | X.XXX | X.XXX |
| F1 | X.XXX | X.XXX |

**Table 3: LLM-as-a-Judge Evaluation**
| Aspect | Baseline | Fine-Tuned |
|--------|----------|------------|
| Correctness | X.X/5.0 | X.X/5.0 |
| Confidence Calibration | X.X/5.0 | X.X/5.0 |
| Reasoning Quality | X.X/5.0 | X.X/5.0 |
| Overall Quality | X.X/5.0 | X.X/5.0 |

---

## ✅ Checklist

### Setup Phase
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Verify GPU availability
- [ ] Download dataset (`native_ads_dataset.xlsx`)

### Data Preparation
- [ ] Run dataset converter
- [ ] Verify output files (QnA, Instruction, Chat formats)
- [ ] Check sample data

### Testing
- [ ] Test single URL with baseline model
- [ ] Verify all 6 agents working
- [ ] Check output format

### Fine-Tuning
- [ ] Quick test with small model (phi-2, 1 epoch)
- [ ] Full fine-tuning (Mistral-7B, 3 epochs)
- [ ] Test fine-tuned model

### Evaluation
- [ ] Evaluate baseline model (100 samples)
- [ ] Evaluate fine-tuned model (100 samples)
- [ ] Full evaluation (1000 samples)
- [ ] Compare results

### Analysis
- [ ] Generate comparison tables
- [ ] Statistical significance testing
- [ ] Error analysis
- [ ] Prepare for publication

---

## 📞 Support

Jika ada masalah, check:
1. `TROUBLESHOOTING_GUIDE.md` - Solusi error umum
2. `FINETUNING_GUIDE.md` - Detail fine-tuning
3. `EVALUATION_GUIDE.md` - Detail evaluasi
4. `RESEARCH_GUIDE.md` - Metodologi penelitian

---

**Good luck dengan penelitian postdoc Anda!** 🎓🚀
