# Agentic AI untuk Deteksi Native Ads pada Portal Berita Elektronik

**Penelitian Postdoc**: Pengembangan sistem Agentic AI untuk mendeteksi native advertising pada portal berita elektronik menggunakan pendekatan multi-agent dengan LLM.

## 📋 Deskripsi Penelitian

Native advertising adalah bentuk iklan yang menyerupai konten editorial, sehingga sulit dibedakan oleh pembaca. Penelitian ini mengembangkan sistem Agentic AI yang dapat:

1. **Scraping otomatis** artikel dari portal berita
2. **Analisis konten** untuk mengidentifikasi karakteristik native ads
3. **Klasifikasi** menggunakan LLM dengan context-aware reasoning
4. **Explainability** untuk memahami keputusan klasifikasi
5. **Continuous learning** dari feedback

## 🏗️ Arsitektur Sistem

```
Portal Berita (URL)
    ↓
Web Agent (Scraping)
    ↓
Preprocessing Agent (Feature Extraction)
    ↓
    ├─→ Retriever Agent (Knowledge Base)
    ├─→ LLM Classifier Agent (Native Ads Detection)
    ├─→ Explanation Agent (Interpretability)
    └─→ Feedback/ReTrainer Agent (Learning)
```

## 📁 Struktur Proyek

```
agentic-ai/
├── main.py                          # Orchestrator utama
├── config.json                      # Konfigurasi sistem
├── requirements.txt                 # Dependencies
│
├── agents/                          # Modular agents
│   ├── web_agent.py                # Web scraping portal berita
│   ├── preprocessing_agent.py      # Text preprocessing & features
│   ├── retriever_agent.py          # Document retrieval (RAG)
│   ├── llm_classifier_agent.py     # LLM-based classification
│   ├── explanation_agent.py        # Explainable AI
│   └── feedback_retrainer_agent.py # Continuous learning
│
├── examples/
│   ├── example_native_ads.py       # Contoh deteksi native ads
│   ├── example_simple.py           # Contoh sederhana
│   └── example_batch.py            # Batch processing
│
└── data/
    ├── feedback/                   # Feedback logs
    └── vector_db/                  # Knowledge base
```

## 🚀 Instalasi

```bash
# 1. Clone/navigate ke direktori
cd "/Users/brianrizqi/Documents/Post Doc/agentic-ai"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup OpenAI API key di config.json
# Edit config.json dan masukkan API key Anda
```

## ⚙️ Konfigurasi

Edit `config.json`:

```json
{
  "llm": {
    "api_key": "sk-your-openai-api-key",
    "model_name": "gpt-4"
  }
}
```

## 📖 Penggunaan

### 1. Deteksi Native Ads pada Artikel Berita

```python
from main import AgenticAISystem
import json

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize system
system = AgenticAISystem(config)

# Scrape dan analisis artikel
url = "https://detik.com/artikel-contoh"
result = system.process_url(url)

# Lihat hasil klasifikasi
print(f"Tipe Konten: {result['classification']['label']}")
print(f"Confidence: {result['classification']['confidence']:.2%}")
print(f"Penjelasan: {result['explanation']['summary']}")
```

### 2. Kategori Klasifikasi

Sistem mengklasifikasikan konten ke dalam kategori:

- **Editorial Content**: Konten berita murni/editorial
- **Native Advertising**: Iklan yang menyerupai konten editorial
- **Sponsored Content**: Konten bersponsor yang jelas ditandai
- **Advertorial**: Iklan dalam format editorial
- **Promotional**: Konten promosi produk/layanan

## 🎯 Fitur Utama

### 1. Web Agent - Portal Berita Scraping
- Scraping otomatis dari berbagai portal berita Indonesia
- Ekstraksi title, content, metadata
- Handle berbagai format HTML

### 2. Preprocessing Agent
- Text cleaning dan normalization
- Feature extraction khusus native ads:
  - Promotional language detection
  - Brand mention frequency
  - Call-to-action patterns
  - Disclosure statement analysis

### 3. Retriever Agent (RAG)
- Knowledge base native ads examples
- Vector similarity search
- Context-aware retrieval

### 4. LLM Classifier Agent
- GPT-4 based classification
- Context dari knowledge base
- Multi-factor analysis

### 5. Explanation Agent
- Interpretable AI decisions
- Key factors identification
- Transparency untuk penelitian

### 6. Feedback/ReTrainer Agent
- Collect feedback dari expert
- Quality metrics
- Export training data

## 📊 Contoh Output

```json
{
  "classification": {
    "label": "Native Advertising",
    "confidence": 0.87,
    "reasoning": "Artikel mengandung promotional language, multiple brand mentions, dan call-to-action tersembunyi"
  },
  "explanation": {
    "summary": "Konten terdeteksi sebagai native advertising...",
    "key_factors": [
      "Penggunaan bahasa promosi: 'terbaik', 'solusi'",
      "Brand mention: 15 kali dalam artikel",
      "Tidak ada disclosure statement",
      "Struktur menyerupai artikel editorial"
    ]
  },
  "features": {
    "promotional_words": 12,
    "brand_mentions": 15,
    "has_disclosure": false,
    "cta_detected": true
  }
}
```

## 🔬 Untuk Penelitian

### Dataset Collection

```python
# Collect dataset dari multiple portal berita
news_portals = [
    "https://detik.com",
    "https://kompas.com",
    "https://tribunnews.com"
]

# Process dan simpan hasil
results = system.process_multiple_urls(urls_from_portals)
```

### Evaluation Metrics

```python
# Get feedback summary untuk evaluasi
stats = system.feedback_agent.get_feedback_summary()

print(f"Accuracy: {stats['avg_quality_score']:.2%}")
print(f"Total samples: {stats['total_feedback']}")
```

### Export Training Data

```python
# Export untuk retraining atau analysis
system.feedback_agent.export_training_data('data/native_ads_dataset.json')
```

## 📚 Komponen Penelitian

### 1. Novelty
- **Multi-agent architecture** untuk native ads detection
- **LLM-based classification** dengan explainability
- **RAG approach** untuk context-aware detection
- **Continuous learning** dari expert feedback

### 2. Metodologi
- Web scraping otomatis dari portal berita
- Feature extraction khusus native advertising
- LLM classification dengan prompt engineering
- Explainable AI untuk interpretability

### 3. Kontribusi
- Framework Agentic AI untuk native ads detection
- Dataset native ads dari portal berita Indonesia
- Comparative analysis: Agentic AI vs traditional ML
- Explainability analysis untuk transparency

## 🎓 Publikasi Potensial

Sistem ini mendukung penelitian untuk publikasi:
- **Jurnal Internasional**: AI, NLP, Digital Marketing
- **Konferensi**: AAAI, ACL, ICML, WWW
- **Topik**: Agentic AI, Native Ads Detection, Explainable AI

## 📝 Citation

```bibtex
@phdthesis{native_ads_agentic_ai,
  title={Pengembangan Agentic AI untuk Deteksi Native Ads pada Portal Berita Elektronik},
  author={[Your Name]},
  year={2024},
  school={[Your University]}
}
```

## 📧 Contact

[Your Contact Information]

## 📄 License

[Your License]
