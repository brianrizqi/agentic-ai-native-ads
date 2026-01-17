# Agentic Orchestrator Chatbot

Intelligent chatbot yang memahami intent user dan otomatis routing ke agent yang tepat menggunakan natural language understanding.

## 🎯 Fitur

**5 Agent Terintegrasi:**
1. **Scraping Agent** - Extract konten dari URL
2. **Preprocessing Agent** - Bersihkan teks, ekstrak fitur, buat ringkasan
3. **Retrieval Agent** - Cari contoh di vector database (RAG)
4. **Classification Agent** - Klasifikasi native ads/berita murni
5. **Reasoning Agent** - Jelaskan detail reasoning

**Capabilities:**
- ✅ **Local intent detection** - Keyword-based, no API needed!
- ✅ Natural language understanding (tidak kaku)
- ✅ Conversation memory (5 pesan terakhir)
- ✅ Multi-step workflows
- ✅ RAG-based classification
- ✅ Optional API usage (hanya untuk classification/explanation)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Chatbot (No API Key Needed!)

**Intent detection is 100% local** - tidak perlu OpenAI API!

```bash
cd langchain-refactor
python chatbot.py
```

**API key hanya diperlukan untuk:**
- Classification Agent (jika pakai OpenAI/OpenRouter)
- Explanation Agent (jika pakai OpenAI/OpenRouter)

**Untuk full local mode:**
```bash
python chatbot.py --provider local --model ../models/qwen-native-ads_merged_16bit
```

**Untuk pakai OpenAI (classification/explanation saja):**
```bash
export OPENAI_API_KEY="your-key"
python chatbot.py --provider openai --api-key $OPENAI_API_KEY
```

## 💬 Contoh Penggunaan

### Scenario 1: Analisis URL Lengkap
```
You: analisis lengkap https://example.com/article

🔍 Intent: full_pipeline
🚀 Running full pipeline...
✅ Analisis lengkap selesai

   📄 Title: Example Article
   🏷️  Label: native ads
   💯 Confidence: 85%
   💭 Reasoning: Konten bersifat promosi produk...
```

### Scenario 2: Multi-Step Workflow
```
You: ambil konten dari https://example.com

🔍 Intent: scrape
✅ Konten berhasil diambil
   📊 Word count: 450

You: sekarang klasifikasikan

🔍 Intent: classify
✅ Klasifikasi: NATIVE ADS
   💯 Confidence: 78%

You: jelaskan kenapa

🔍 Intent: explain
✅ Penjelasan detail:
   [Detailed explanation...]
```

### Scenario 3: Direct Classification
```
You: klasifikasikan: "Dapatkan diskon 50% di Tokopedia hari ini!"

🔍 Intent: classify
✅ Klasifikasi: NATIVE ADS
   💯 Confidence: 92%
```

### Scenario 4: Search Examples
```
You: cari contoh native ads tentang teknologi

🔍 Intent: retrieve
✅ Ditemukan 5 contoh

   [1] NATIVE ADS
       Promosi smartphone terbaru...
       Similarity: 0.92
```

## 🎮 Commands

- `/help` - Tampilkan bantuan
- `/clear` - Hapus history percakapan
- `/memory` - Lihat history percakapan
- `/exit` - Keluar (otomatis save conversation)

## 🏗️ Architecture

```
User Input
    ↓
Intent Detector (LLM)
    ↓
Orchestrator Agent
    ↓
┌─────────────────────────────────┐
│ 1. Scraping Agent               │
│ 2. Preprocessing Agent          │
│ 3. Retrieval Agent (RAG)        │
│ 4. Classification Agent         │
│ 5. Reasoning Agent              │
└─────────────────────────────────┘
    ↓
Formatted Response
```

## 📁 File Structure

```
langchain-refactor/
├── agents/
│   ├── preprocessing_agent.py    # NEW: Text preprocessing
│   ├── retrieval_agent.py        # NEW: RAG retrieval
│   ├── intent_detector.py        # NEW: Intent detection
│   ├── orchestrator_agent.py     # NEW: Main orchestrator
│   ├── classification_agent.py   # Existing
│   └── explanation_agent.py      # Existing
├── chatbot.py                     # NEW: Interactive CLI
└── ...
```

## 🔧 Advanced Usage

### Use with Local Fine-Tuned Model

```bash
python chatbot.py \
  --provider local \
  --model ../models/qwen-native-ads_merged_16bit
```

### Use with OpenRouter

```bash
python chatbot.py \
  --provider openrouter \
  --model google/gemini-pro \
  --api-key $OPENROUTER_API_KEY
```

### Disable Retrieval (No Vector DB)

```bash
python chatbot.py --no-vectorstore
```

## 🧪 Testing

Test individual agents:

```python
from agents import OrchestratorAgent

orchestrator = OrchestratorAgent(
    model_name="gpt-3.5-turbo",
    provider="openai",
    api_key="your-key"
)

# Test intent detection
response = orchestrator.process("klasifikasikan berita ini")
print(response)
```

## 📝 Notes

- **RAG Architecture**: Retrieval sebelum Classification untuk accuracy lebih baik
- **Conversation Memory**: Menyimpan 5 pesan terakhir untuk context
- **Auto-save**: Conversation otomatis disave saat exit
- **Flexible Input**: Mendukung bahasa Indonesia dan English

## 🐛 Troubleshooting

**Error: "No vectorstore available"**
- Retrieval features disabled
- Run with `--no-vectorstore` atau setup vector database

**Error: "API key not found"**
- Set environment variable atau gunakan `--api-key`

**Intent detection tidak akurat**
- Gunakan model yang lebih baik (gpt-4)
- Atau gunakan keyword fallback (otomatis)

## 📚 Related Files

- [Implementation Plan](../brain/implementation_plan.md)
- [Main Pipeline](main.py) - Original CLI
- [Fine-tuning Guide](FINETUNING.md)
