# Dataset Conversion Guide

Panduan lengkap untuk mengkonversi dataset Deep Learning (konten + label) menjadi format LLM.

## 📋 Overview

Dataset Anda saat ini:
- **Format**: Excel (native_ads_dataset.xlsx)
- **Struktur**: Kolom konten + kolom label
- **Use case**: Deep Learning classification

Target konversi:
- **Format LLM**: QnA, Instruction-Response, Chat
- **Use case**: LLM fine-tuning, RAG, prompt engineering

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pandas openpyxl groq ollama
```

### 2. Konversi Dasar (Tanpa LLM)

```bash
cd tools
python dataset_converter.py
```

Script akan:
- Load `native_ads_dataset.xlsx`
- Generate 3 format output:
  - `llm_dataset_qna.json` - QnA format
  - `llm_dataset_instruction.jsonl` - Instruction format
  - `llm_dataset_chat.jsonl` - Chat format (OpenAI)

### 3. Augmentasi dengan LLM (Opsional)

```bash
python llm_augmentation.py
```

Pilih provider:
- **Groq**: Gratis, cepat (perlu API key dari console.groq.com)
- **Ollama**: Local, gratis (perlu install Ollama)

## 📊 Format Output

### 1. QnA Format

Cocok untuk: RAG systems, question-answering

```json
{
  "id": "sample_0",
  "question": "Apakah artikel berikut ini termasuk native advertising?",
  "context": "[konten artikel...]",
  "answer": "Ya, ini adalah Native Advertising. Konten ini...",
  "label": "native_ads"
}
```

### 2. Instruction Format

Cocok untuk: Fine-tuning LLM (Llama, Mistral, Gemma)

```json
{
  "id": "sample_0",
  "instruction": "Klasifikasikan artikel berikut sebagai native ads atau editorial.",
  "input": "[konten artikel...]",
  "output": "**Klasifikasi**: Native Advertising\n\n**Analisis**:..."
}
```

### 3. Chat Format

Cocok untuk: Fine-tuning ChatGPT, Claude

```json
{
  "messages": [
    {"role": "system", "content": "Anda adalah expert deteksi native ads..."},
    {"role": "user", "content": "Klasifikasikan artikel: [konten]"},
    {"role": "assistant", "content": "Klasifikasi: Native Advertising..."}
  ]
}
```

## 🎯 Use Cases

### Use Case 1: Fine-tuning LLM Lokal

```bash
# 1. Konversi dataset
python dataset_converter.py

# 2. Fine-tune dengan Ollama
ollama create native-ads-detector -f Modelfile

# 3. Test model
ollama run native-ads-detector "Klasifikasikan artikel ini: ..."
```

### Use Case 2: RAG System

```python
# Load QnA dataset
with open('data/llm_dataset_qna.json') as f:
    qna_data = json.load(f)

# Add ke vector database
for item in qna_data:
    retriever_agent.add_documents([item['context']])
```

### Use Case 3: Prompt Engineering

```python
# Load instruction dataset
with open('data/llm_dataset_instruction.jsonl') as f:
    examples = [json.loads(line) for line in f]

# Few-shot prompting
prompt = "Berikut contoh klasifikasi:\n\n"
for ex in examples[:3]:
    prompt += f"Input: {ex['input'][:100]}\n"
    prompt += f"Output: {ex['output']}\n\n"
prompt += f"Sekarang klasifikasikan: {new_article}"
```

## 🆓 LLM Gratis untuk Augmentasi

### 1. Groq (Recommended)

**Kelebihan**:
- Gratis unlimited
- Sangat cepat (inference time < 1s)
- Model bagus: Mixtral, Llama 3

**Setup**:
```bash
# 1. Daftar di console.groq.com
# 2. Get API key
# 3. Run augmentation
python llm_augmentation.py
# Pilih option 1 (Groq)
```

### 2. Ollama (Local)

**Kelebihan**:
- 100% gratis
- Privacy (local)
- Banyak model: Llama 2, Mistral, Gemma

**Setup**:
```bash
# 1. Install Ollama
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull model
ollama pull llama2

# 3. Run augmentation
python llm_augmentation.py
# Pilih option 2 (Ollama)
```

### 3. HuggingFace (Colab)

**Kelebihan**:
- Gratis dengan GPU
- Banyak model open-source

**Setup**:
```python
# Di Google Colab
!pip install transformers torch

from transformers import pipeline
generator = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")
```

## 📈 Data Augmentation Strategies

### Strategy 1: Question Variation

Generate 3-5 variasi pertanyaan per sample:
- Direct: "Apakah ini native ads?"
- Indirect: "Analisis jenis konten ini"
- Specific: "Identifikasi indikator native advertising"

### Strategy 2: Explanation Enhancement

LLM generate penjelasan detail:
- Reasoning chain
- Evidence dari teks
- Confidence level
- Alternative interpretations

### Strategy 3: Synthetic Examples

Generate contoh baru berdasarkan pola:
- Paraphrase konten
- Variasi label
- Edge cases

## 🔧 Customization

### Custom Question Templates

Edit di `dataset_converter.py`:

```python
question_templates = [
    "Pertanyaan custom 1",
    "Pertanyaan custom 2",
    # Tambahkan template Anda
]
```

### Custom Label Mapping

```python
label_explanations = {
    'native_ads': 'Penjelasan custom untuk native ads',
    'editorial': 'Penjelasan custom untuk editorial',
    # Tambahkan mapping Anda
}
```

## 📊 Quality Check

### Validasi Output

```python
import json

# Load converted data
with open('data/llm_dataset_qna.json') as f:
    data = json.load(f)

# Check
print(f"Total samples: {len(data)}")
print(f"Sample: {data[0]}")

# Validate structure
for item in data:
    assert 'question' in item
    assert 'context' in item
    assert 'answer' in item
```

### Manual Review

Review beberapa sample secara manual:
1. Apakah pertanyaan relevan?
2. Apakah jawaban akurat?
3. Apakah penjelasan masuk akal?

## 🎓 Untuk Penelitian

### Dataset Splitting

```python
from sklearn.model_selection import train_test_split

# Split 80-10-10
train, temp = train_test_split(data, test_size=0.2)
val, test = train_test_split(temp, test_size=0.5)

# Save
with open('train.jsonl', 'w') as f:
    for item in train:
        f.write(json.dumps(item) + '\n')
```

### Evaluation Metrics

Untuk fine-tuned model:
- Accuracy
- Precision/Recall per class
- F1-score
- Confusion matrix
- Confidence calibration

## 💡 Tips

1. **Start Simple**: Mulai dengan konversi dasar dulu
2. **Validate**: Selalu cek output secara manual
3. **Iterate**: Improve templates berdasarkan hasil
4. **Augment**: Gunakan LLM untuk variasi lebih banyak
5. **Document**: Catat semua perubahan untuk reproducibility

## 🐛 Troubleshooting

### Error: File not found
```bash
# Pastikan file ada
ls native_ads_dataset.xlsx

# Atau specify path lengkap
python dataset_converter.py --input /path/to/file.xlsx
```

### Error: Column not found
```bash
# Check column names
python -c "import pandas as pd; print(pd.read_excel('native_ads_dataset.xlsx').columns)"
```

### Error: Groq API
```bash
# Check API key
echo $GROQ_API_KEY

# Atau input manual saat runtime
```

## 📚 References

- [Groq Documentation](https://console.groq.com/docs)
- [Ollama Documentation](https://ollama.com/docs)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
- [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)

---

**Last Updated**: 2024
**Author**: [Your Name]
