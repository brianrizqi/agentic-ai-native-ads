# Agentic AI untuk Deteksi Native Ads pada Portal Berita Elektronik

**Penelitian Postdoc**: Pengembangan sistem Agentic AI untuk mendeteksi native advertising pada portal berita elektronik menggunakan pendekatan multi-agent dengan LLM (Large Language Model).

## 📋 Deskripsi Penelitian

Native advertising adalah bentuk iklan yang menyerupai konten editorial, sehingga sulit dibedakan oleh pembaca. Penelitian ini mengembangkan sistem Agentic AI yang dapat:

1.  **Scraping otomatis** artikel dari portal berita
2.  **Analisis konten** untuk mengidentifikasi karakteristik native ads
3.  **Klasifikasi** menggunakan LLM dengan context-aware reasoning
4.  **Explainability** untuk memahami keputusan klasifikasi
5.  **Continuous learning** dari feedback

## 🏗️ Arsitektur Sistem

```mermaid
graph TD
    User[User/Researcher] -->|Input URL| WebAgent
    WebAgent -->|Scraped Data| PreprocessingAgent
    PreprocessingAgent -->|Cleaned Text| Pipeline
    
    subgraph Pipeline [Agentic AI Pipeline]
        RetrieverAgent -->|Context Docs| LLMClassifierAgent
        LLMClassifierAgent -->|Classification| ExplanationAgent
        ExplanationAgent -->|Explanation| Result
    end
    
    Dataset[(Native Ads Dataset)] -->|Vector DB| RetrieverAgent
    LocalLLM[Local LLM (Ollama/HF)] --> LLMClassifierAgent
    LocalLLM --> ExplanationAgent
```

## 📁 Struktur Proyek

```
agentic-ai/
├── main.py                          # Orchestrator utama
├── run_local_demo.py                # Script demo (Local LLM)
├── config.json                      # Konfigurasi sistem
├── requirements.txt                 # Dependencies
│
├── agents/                          # Modular agents
│   ├── web_agent.py                # Web scraping
│   ├── preprocessing_agent.py      # Text preprocessing
│   ├── retriever_agent.py          # RAG Retrieval
│   ├── llm_classifier_agent.py     # Classification
│   ├── explanation_agent.py        # Explainable AI
│   └── ...
│
├── tools/
│   ├── dataset_converter.py        # Konversi Excel -> JSON untuk RAG
│   └── llm_augmentation.py         # Augmentasi dataset
│
└── data/
    ├── native_ads_dataset.xlsx     # Raw dataset
    └── llm_dataset_qna.json        # Converted knowledge base
```

## 🚀 Instalasi & Persiapan

### 1. Prasyarat Sistem
-   **Python 3.10+**
-   **Ollama** (untuk menjalankan LLM lokal seperti Llama 3 / Mistral)
-   *Optional*: GPU NVIDIA dengan CUDA untuk performa lebih baik (jika menggunakan HuggingFace provider).

### 2. Setup Lingkungan
Disarankan menggunakan virtual environment:

```bash
# Buat virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  # Jika pakai GPU
pip install ollama
```

### 4. Setup Ollama (Wajib untuk Local LLM)
1.  Download & Install [Ollama](https://ollama.com/)
2.  Pull model yang akan digunakan (contoh: `llama3` atau `mistral`):
    ```bash
    ollama pull llama3
    ```
3.  Jalankan service:
    ```bash
    ollama serve
    ```

---

## 📖 Quick Start Guide (Tutorial)

Berikut urutan langkah untuk menjalankan sistem dari awal hingga eksperimen.

### Langkah 1: Konversi Dataset
Kita perlu mengubah dataset mentah (Excel) menjadi format JSON yang bisa dibaca oleh Retriever Agent (RAG).

1.  Pastikan file dataset ada di folder root atau `tools/`.
2.  Jalankan script konversi:
    ```bash
    python tools/dataset_converter.py
    ```
3.  Ikuti prompt di terminal:
    -   Masukkan path file Excel: `native_ads_dataset.xlsx`
    -   Pilih format output: `1` (QnA / Knowledge Base format)
    -   Isi nama kolom (jika diminta, default usually OK).
4.  **Hasil**: File `data/llm_dataset_qna.json` akan terbentuk. Ini adalah "otak" atau knowledge base untuk sistem.

### Langkah 2: Menjalankan Eksperimen (Pipeline Executor)
Gunakan script `run_local_demo.py` untuk menjalankan pipeline lengkap pada satu artikel berita.

**Command Dasar (Menggunakan Ollama):**
```bash
python run_local_demo.py --url "https://news-url.com/artikel" --provider ollama --model llama3
```

**Command Menggunakan HuggingFace (GPU):**
```bash
python run_local_demo.py --url "https://news-url.com/artikel" --provider huggingface --model gpt2
```

**Penjelasan Parameter:**
-   `--url`: URL artikel berita yang ingin dianalisis.
-   `--provider`: Backend LLM (`ollama` atau `huggingface`).
-   `--model`: Nama model (sesuai yang di-pull di Ollama atau nama repo HF).

### Langkah 3: Membaca Hasil
Script akan menampilkan log proses di terminal:
1.  **Scraping**: Judul dan panjang teks yang diambil.
2.  **Preprocessing**: Tokenisasi dan cleaning.
3.  **Retrieval**: Dokumen mirip yang ditemukan di dataset (Context).
4.  **Classification**: Hasil deteksi (Native Ads vs Editorial) beserta confidence score.
5.  **Explanation**: Penjelasan mengapa konten tersebut diklasifikasikan demikian.

Hasil lengkap juga disimpan otomatis ke file `local_demo_result.json`.

---

## 🎯 Komponen Utama

### 1. Web Agent (`web_agent.py`)
Bertugas mengambil konten mentah HTML dari URL dan mengekstrak teks utama, judul, dan metadata.

### 2. Retriever Agent (`retriever_agent.py`)
Menggunakan **Sentence Transformers** untuk mengubah teks artikel menjadi vektor, lalu mencari artikel serupa di database (`llm_dataset_qna.json`) sebagai konteks bagi LLM.

### 3. LLM Classifier & Explanation Agent
Menggunakan Local LLM untuk "berpikir".
-   **Classifier**: Memutuskan label (Native Ads/Editorial) berdasarkan konten dan konteks retrieval.
-   **Explainer**: Menulis paragraf penjelasan yang mudah dipahami manusia.

## ⚠️ Troubleshooting

-   **Error `c10.dll` / Torch**: Install Microsoft Visual C++ Redistributable atau pastikan versi Torch sesuai dengan CUDA system Anda.
-   **Ollama Connection Error**: Pastikan aplikasi Ollama sudah berjalan (`ollama serve`).
-   **UnicodeEncodeError**: Pada Windows, terminal kadang gagal print karakter emoji (✓). Script otomatis menangani ini, tapi jika masih muncul, set `chcp 65001` di terminal.

## 📧 Kontak & Sitasi
[Informasi Peneliti / Lab]
