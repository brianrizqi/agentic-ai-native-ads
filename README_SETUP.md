# Setup Guide - Agentic AI Native Ads Detection

Quick setup guide untuk penelitian postdoc Anda.

## ✅ Yang Sudah Selesai

1. **Dataset Conversion** ✓
   - 12,088 samples berhasil dikonversi
   - 3 format: QnA, Instruction, Chat
   - Files: `data/llm_dataset_*.json/jsonl`

2. **Project Structure** ✓
   - Modular agents (6 agents terpisah)
   - Tools untuk dataset processing
   - Examples & documentation

## 🚀 Quick Start (Ringan)

### 1. Setup Virtual Environment

```bash
cd "/Users/brianrizqi/Documents/Post Doc/agentic-ai"
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies (Minimal)

Untuk laptop yang tidak kuat, install minimal saja:

```bash
# Core dependencies only
pip install pandas openpyxl requests beautifulsoup4 numpy
```

### 3. Konversi Dataset (Sudah Selesai)

Dataset sudah dikonversi! Files ada di:
- `data/llm_dataset_qna.json` (14MB)
- `data/llm_dataset_instruction.jsonl` (14MB)
- `data/llm_dataset_chat.jsonl` (17MB)

## 📊 Gunakan Dataset

### Option 1: Upload ke Cloud untuk Fine-tuning

**Google Colab (Gratis dengan GPU)**:
```python
# Upload file ke Colab
from google.colab import files
uploaded = files.upload()  # Upload llm_dataset_instruction.jsonl

# Fine-tune dengan HuggingFace
from transformers import AutoModelForSequenceClassification, Trainer
# ... training code
```

**Groq (Gratis, Cepat)**:
- Daftar di https://console.groq.com
- Upload dataset untuk inference (bukan fine-tuning)
- Gunakan untuk augmentasi data

### Option 2: Analisis Dataset Lokal (Ringan)

```python
import json
import pandas as pd

# Load dataset
with open('data/llm_dataset_qna.json') as f:
    data = json.load(f)

# Analisis
df = pd.DataFrame(data)
print(df['label'].value_counts())

# Export ke CSV untuk analisis
df.to_csv('dataset_analysis.csv', index=False)
```

### Option 3: Gunakan untuk RAG (Tanpa Training)

Gunakan dataset sebagai knowledge base untuk RAG system tanpa perlu training model berat.

## 💡 Rekomendasi untuk Laptop Tidak Kuat

### 1. Jangan Install Dependencies Berat

Skip ini jika laptop lemah:
```bash
# JANGAN install ini:
# sentence-transformers (90MB+ model)
# torch (besar)
# transformers (besar)
```

### 2. Gunakan Cloud Services

**Untuk Fine-tuning**:
- Google Colab (gratis, GPU)
- Kaggle Notebooks (gratis, GPU)
- HuggingFace Spaces (gratis)

**Untuk Inference**:
- Groq API (gratis, cepat)
- Ollama (local tapi ringan)
- OpenAI API (berbayar tapi powerful)

### 3. Workflow Recommended

```
Local (Laptop Anda):
├── Dataset conversion ✓ (sudah selesai)
├── Data analysis (pandas, ringan)
└── Code development

Cloud (Colab/Kaggle):
├── Model training
├── Fine-tuning
└── Evaluation
```

## 📁 File Structure

```
agentic-ai/
├── data/
│   ├── llm_dataset_qna.json          (14MB) ✓
│   ├── llm_dataset_instruction.jsonl (14MB) ✓
│   └── llm_dataset_chat.jsonl        (17MB) ✓
│
├── agents/                    (Modular agents)
├── tools/                     (Dataset tools)
├── examples/                  (Usage examples)
│
├── native_ads_dataset.xlsx    (Dataset asli)
├── requirements.txt           (Dependencies)
├── config.json               (Configuration)
└── README.md                 (Main documentation)
```

## 🎓 Next Steps untuk Penelitian

### Phase 1: Data Analysis (Lokal, Ringan)
```bash
# Analisis dataset dengan pandas
python -c "import pandas as pd; import json; data = json.load(open('data/llm_dataset_qna.json')); df = pd.DataFrame(data); print(df.describe())"
```

### Phase 2: Fine-tuning (Cloud)
Upload `llm_dataset_instruction.jsonl` ke Google Colab dan fine-tune model seperti:
- Llama 2 7B
- Mistral 7B
- Gemma 7B

### Phase 3: Evaluation (Cloud)
Test model dengan test set dan hitung metrics.

### Phase 4: Deployment
Deploy model untuk production testing.

## 🔧 Git Setup

```bash
# Initialize git
git init
git add .
git commit -m "Initial commit: Agentic AI Native Ads Detection"

# Push ke GitHub
git remote add origin <your-repo-url>
git push -u origin main
```

## 📝 Files yang Sudah Siap Pakai

1. **Dataset** (3 format, 12,088 samples each)
2. **Conversion tools** (`tools/dataset_converter.py`)
3. **Documentation** (README, guides)
4. **Agent code** (6 modular agents)

## ⚠️ Catatan Penting

- Dataset files (45MB total) sudah di-gitignore
- Virtual environment (`venv/`) sudah di-gitignore
- Model cache sudah di-gitignore
- Hanya code yang akan di-commit ke Git

## 🆘 Troubleshooting

**Laptop lemot saat run script?**
- Jangan run `dataset_executor.py` (butuh model berat)
- Gunakan cloud untuk testing
- Fokus ke data analysis dengan pandas saja

**Out of memory?**
- Jangan load semua 12K samples sekaligus
- Gunakan chunking: `data[:1000]` untuk 1000 samples pertama
- Process batch by batch

**Model download lambat?**
- Skip sentence-transformers
- Gunakan Groq API (cloud-based, gratis)
- Atau Ollama (local tapi optional)

---

**Kesimpulan**: Dataset sudah siap! Untuk laptop yang tidak kuat, gunakan cloud services (Colab/Groq) untuk training dan inference. Local hanya untuk data analysis dan code development.
