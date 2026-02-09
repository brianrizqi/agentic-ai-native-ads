# Native Ads Detection - LangChain Refactor

Sistem deteksi native ads menggunakan LangChain dengan multiple agents dan fine-tuned local models.

---

## 🎯 Overview

Project ini mengklasifikasikan artikel berita sebagai **native ads** atau **berita murni** menggunakan:
- **Multi-agent orchestration** (Scraping, Preprocessing, Retrieval, Classification, Reasoning)
- **Fine-tuned local LLMs** (Qwen, Llama, Gemma)
- **RAG pipeline** untuk context-aware classification

**Hasil Terbaik:** 97% accuracy dengan Qwen 2.5 14B fine-tuned model
- **Advanced Insights:** Brand detection, Techniques, Sentiment, and CTA via Instructor Mode.

---

## 📁 Project Structure

```
langchain-refactor/
├── agents/                      # Agent implementations
│   ├── classification_agent.py  # Main classification logic
│   ├── instructor_classification_agent.py # Advanced analysis with Instructor
│   ├── orchestrator_agent.py    # Multi-agent coordinator
│   ├── scraping_agent.py        # Web scraping
│   ├── preprocessing_agent.py   # Text preprocessing
│   ├── retrieval_agent.py       # RAG retrieval
│   └── reasoning_agent.py       # Explanation generation
├── prompts/                     # Prompt templates
│   └── classification_prompts.py
├── tools/                       # Utility scripts
│   ├── prepare_finetuning_dataset.py
│   └── clean_dataset.py
├── eval_results/                # Evaluation outputs
├── finetune.py                  # Main fine-tuning script
├── evaluate_model.py            # Model evaluation
├── chatbot.py                   # Interactive CLI
├── main.py                      # Single URL classification
└── compare_models.py            # Multi-model comparison
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
cd langchain-refactor

# Install dependencies
pip install -r requirements.txt

# For fine-tuning (GPU required)
pip install -r requirements-finetuning.txt

# For evaluation
pip install -r requirements-evaluation.txt

# For Instructor Mode (Advanced Analysis)
pip install instructor pydantic
```

### 2. Basic Usage

**Classify a single URL:**
```bash
python3 main.py --url "https://example.com/article"
```

**Interactive chatbot:**
```bash
python3 chatbot.py
```

**With local fine-tuned model:**
```bash
python3 chatbot.py \
  --provider local \
  --model ../models/qwen-native-ads-v2_merged_16bit

# Advanced Analysis (Instructor Mode)
python3 chatbot.py --use-instructor
```

*Gunakan mode ini untuk mendapatkan Brand, Teknik Persuasi, Sentiment, dan CTA.*

---

## 🤖 Fine-Tuning

### Prerequisites
- GPU with 24GB+ VRAM (A100 recommended)
- CUDA 11.8+ or 12.1+
- 50GB+ free disk space

### Step 1: Preprocess Dataset

```bash
python3 tools/prepare_finetuning_dataset.py \
  --input ../data/llm_dataset_detailed.json \
  --output ../data/llm_dataset_finetuning_optimized.json \
  --max-reasoning 150
```

### Step 2: Clean Dataset (Optional but Recommended)

```bash
python3 tools/clean_dataset.py \
  --input ../data/llm_dataset_finetuning_optimized.json \
  --output ../data/llm_dataset_clean.json
```

### Step 3: Fine-Tune

**Qwen 2.5 14B (Recommended):**
```bash
python3 finetune.py \
  --model qwen \
  --dataset ../data/llm_dataset_clean.json \
  --output ../models/qwen-native-ads-v4 \
  --max-steps 1500
```

**Llama 3.1 8B (Faster):**
```bash
python3 finetune.py \
  --model llama \
  --dataset ../data/llm_dataset_clean.json \
  --output ../models/llama-native-ads-v4 \
  --max-steps 1500
```

**Gemma 2 9B:**
```bash
python3 finetune.py \
  --model gemma \
  --dataset ../data/llm_dataset_clean.json \
  --output ../models/gemma-native-ads-v4 \
  --max-steps 1500
```

**GPT OSS:**
```bash
python3 finetune_gpt_oss.py \
  --dataset ../data/llm_dataset_clean.json \
  --output ../models/gpt-oss-native-ads-v4 \
  --max-steps 1500
```

**Training time:** ~4-6 hours on A100 (1500 steps)

---

## 📊 Evaluation

### Basic Evaluation

```bash
python3 evaluate_model.py \
  --model ../models/qwen-native-ads-v4_merged_16bit \
  --num-samples 200
```

**Output:**
```
eval_results/
├── eval_qwen_native_ads_v4_20260118_120000.json
├── eval_qwen_native_ads_v4_20260118_120000_summary.txt
└── eval_qwen_native_ads_v4_20260118_120000_confusion_matrix.png
```

### Compare Multiple Models

```bash
python3 compare_models.py \
  --dataset ../data/llm_dataset_clean.json \
  --num-samples 200
```

---

## 📈 Performance

| Model | Accuracy | F1-Score | JSON Parse | Training Time |
|-------|----------|----------|------------|---------------|
| **Qwen 2.5 14B V2** | **97.0%** | **~0.97** | **92.5%** | ~5 hours |
| Llama 3.1 8B V3 | 66.0% | ~0.66 | ~85% | ~3 hours |
| Gemma 2 9B | TBD | TBD | TBD | ~3 hours |

**Best Model:** Qwen 2.5 14B V2 - Use for production!

---

## 🛠️ Available Models

### Supported Base Models

1. **Qwen 2.5 14B** (Recommended)
   - Best multilingual performance
   - Excellent Bahasa Indonesia support
   - 97% accuracy achieved

2. **Llama 3.1 8B**
   - Faster training/inference
   - Good for development/testing
   - 66% accuracy (needs improvement)

3. **Gemma 2 9B**
   - Google's efficient model
   - Alternative option
   - Not yet fully tested

### Model Configurations

All models use optimized LoRA settings:
- `max_seq_length`: 1024
- `lora_r`: 16
- `lora_alpha`: 32
- `learning_rate`: 2e-5
- Batch size: 1-2 with gradient accumulation

---

## 🧠 Instructor Mode (Advanced Analysis)

Instructor Mode meningkatkan agent klasifikasi standar dengan kemampuan ekstraksi entitas terstruktur.

### Extra Features:
- **🏷️ Target Brand:** Menemukan merk/instansi yang dipromosikan.
- **🎯 Techniques:** Mendeteksi teknik (Testimoni, Emotional Language, dll).
- **🎭 Sentiment:** Menganalisis nada (Positif, Negatif, Netral).
- ** CTA Text:** Menangkap ajakan bertindak (Daftar sekarang, Beli, dll).

### Full Usage Example:
1. **Install Dependencies:**
   ```bash
   pip install instructor pydantic
   ```

2. **Run with Local Fine-Tuned Model:**
   ```bash
   python3 chatbot.py \
     --use-instructor \
     --provider local \
     --model ../models/qwen-native-ads-v2_merged_16bit
   ```

3. **Sample Command & Output:**
   *Input:* `jelaskan https://travel.detik.com/...`
   
   *Output:*
   ```text
   ✅ Analisis lengkap selesai
      📝 Summary: Pesawat AirAsia melakukan pendaratan darurat...
      🏷️ Label: berita murni
      💯 Confidence: 95.00%
      💭 Reasoning: Artikel menyajikan fakta secara netral tanpa promosi.
      🎯 Techniques: bahasa emosional
      🎭 Sentiment: netral
   
   Analisis Pakar:
   Artikel dikategorikan berita murni karena nada bicaranya objektif, 
   tidak menyebutkan brand tertentu selain maskapai dalam konteks insiden, 
   dan menyajikan fakta seimbang dari sisi kronologi pendaratan darurat.
   ```

---

## 📚 Documentation

- **[FINETUNING.md](FINETUNING.md)** - Complete fine-tuning guide
- **[FINETUNING_QUICKSTART.md](FINETUNING_QUICKSTART.md)** - Quick reference
- **[EVALUATION_RESULTS.md](EVALUATION_RESULTS.md)** - Evaluation organization
- **[CHATBOT.md](CHATBOT.md)** - Chatbot usage guide

---

## 🔧 Configuration

### Environment Variables

```bash
# OpenAI API (optional)
export OPENAI_API_KEY="your-key"

# OpenRouter API (optional)
export OPENROUTER_API_KEY="your-key"

# For LLM-as-a-Judge evaluation
export OPENROUTER_API_KEY="your-key"
```

### Model Paths

Fine-tuned models are saved in:
```
../models/
├── qwen-native-ads-v2/              # LoRA adapters
├── qwen-native-ads-v2_merged_16bit/ # Production model (use this!)
├── llama-native-ads-v3/
└── llama-native-ads-v3_merged_16bit/
```

**Always use `_merged_16bit` models for inference!**

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Use smaller model
python3 finetune.py --model llama

# Or reduce batch size (edit finetune.py)
```

### Low Accuracy (<80%)
- Ensure dataset is clean (use `clean_dataset.py`)
- Increase training steps to 2000-2500
- Check training curves for overfitting

### JSON Parsing Errors
- Increase `max_new_tokens` in `classification_agent.py`
- Ensure reasoning is truncated in training data
- Use V2 prompt format (simpler is better)

---

## 🎓 Key Learnings

1. **Prompt Consistency** - Training and inference must use identical prompts
2. **Output Length** - Truncate reasoning to 100-150 chars for best results
3. **Clean Data** - Remove error samples before training
4. **Simple Prompts** - Extra instructions can confuse the model

---

## 📞 Support

For issues or questions:
1. Check [FINETUNING.md](FINETUNING.md) troubleshooting section
2. Review evaluation results in `eval_results/`
3. Compare with baseline performance (97% accuracy target)

---

## 📝 License

[Your License Here]

---

**Last Updated:** 2026-01-18  
**Version:** 4.0 (with clean dataset and V2 prompt)
