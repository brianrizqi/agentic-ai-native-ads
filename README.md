# Agentic AI untuk Deteksi Native Ads

**Penelitian Postdoc**: Sistem AI untuk mendeteksi iklan terselubung (native ads) di portal berita online menggunakan teknologi Agentic AI.

---

## 📖 Apa itu Project Ini?

Project ini adalah sistem AI yang bisa **membedakan artikel berita asli dengan iklan yang menyamar sebagai berita** (native ads). Sistem ini menggunakan beberapa "agen AI" yang bekerja sama seperti tim detektif untuk menganalisis artikel berita.

### Contoh Kasus:
- ✅ **Berita Asli**: "Inflasi Bulan Ini Naik 2%, BI Pertimbangkan Naikkan Suku Bunga"
- ❌ **Native Ads**: "Promo Spesial BRI di HUT ke-128, Diskon Hingga Rp1,28 Juta!" ← Ini iklan yang ditulis seperti berita

---

## 🎯 Cara Kerja Sistem (Sederhana)

Sistem ini punya 5 "agen AI" yang bekerja berurutan:

```
1. Web Scraper → Ambil artikel dari URL
2. Preprocessor → Bersihkan teks
3. Retriever → Cari contoh serupa di database
4. Classifier → Tentukan: Native Ads atau Berita Asli?
5. Explainer → Jelaskan alasannya
```

**Analogi**: Seperti tim dokter yang memeriksa pasien:
- Dokter 1: Ambil data pasien
- Dokter 2: Bersihkan data
- Dokter 3: Cek riwayat kasus serupa
- Dokter 4: Diagnosis
- Dokter 5: Jelaskan diagnosis ke pasien

---

## 🚀 Cara Menjalankan (Step-by-Step)

### Langkah 1: Persiapan Awal

**Install Python dan Library:**
```bash
# 1. Buat virtual environment
python -m venv .venv

# 2. Aktifkan virtual environment
# Untuk Mac/Linux:
source .venv/bin/activate
# Untuk Windows:
.venv\Scripts\activate

# 3. Install semua library yang dibutuhkan
pip install -r requirements.txt
```

---

### Langkah 2: Siapkan Dataset

Dataset adalah "buku pengetahuan" yang dipakai AI untuk belajar membedakan berita asli vs iklan.

**Jalankan converter:**
```bash
python tools/dataset_converter.py
```

**Apa yang terjadi?**
- File `native_ads_dataset.xlsx` (Excel) akan diubah jadi format JSON
- Hasilnya: `data/llm_dataset_qna.json` (database pengetahuan AI)

---

### Langkah 3: Jalankan Deteksi

**Command untuk analisis artikel:**
```bash
python run_cpu_mode.py --url "MASUKKAN_URL_ARTIKEL_DISINI"
```

**Contoh nyata:**
```bash
python run_cpu_mode.py --url "https://www.cnnindonesia.com/ekonomi/20231212140523-532-1036329/promo-spesial-bri-di-hut-ke-128-diskon-hingga-rp1-28-juta"
```

**Output yang akan muncul:**
```
CLASSIFICATION RESULT:
Label: Native Ads
Confidence: 92%
Reasoning: Artikel ini mengandung promosi produk, call-to-action, dan bahasa marketing
```

---

## 📂 Struktur Folder Project

```
agentic-ai-native-ads/
│
├── agents/                          # Folder berisi semua agen AI
│   ├── web_agent.py                # Agen 1: Scraping artikel
│   ├── preprocessing_agent.py      # Agen 2: Bersihkan teks
│   ├── retriever_agent.py          # Agen 3: Cari contoh serupa
│   ├── llm_classifier_agent.py     # Agen 4: Klasifikasi
│   └── explanation_agent.py        # Agen 5: Penjelasan
│
├── tools/                           # Tools bantu
│   └── dataset_converter.py        # Ubah Excel → JSON
│
├── data/                            # Folder data
│   ├── native_ads_dataset.xlsx     # Dataset asli (Excel)
│   └── llm_dataset_qna.json        # Dataset untuk AI (JSON)
│
├── run_cpu_mode.py                 # Script utama (CPU mode)
├── requirements.txt                # Daftar library Python
└── README.md                       # File ini
```

---

## 🔧 Troubleshooting (Kalau Ada Masalah)

### Problem 1: "Dataset not found"
**Solusi**: Jalankan dulu converter dataset
```bash
python tools/dataset_converter.py
```

### Problem 2: "Out of Memory"
**Solusi**: Gunakan model yang lebih kecil
```bash
python run_cpu_mode.py --model microsoft/phi-2 --url "URL_ANDA"
```

### Problem 3: "CUDA Error"
**Solusi**: Gunakan CPU mode (sudah otomatis di `run_cpu_mode.py`)

---

## 📊 Hasil Penelitian

Sistem ini diharapkan mencapai:
- ✅ **Akurasi**: >90%
- ✅ **Precision**: >85%
- ✅ **Recall**: >85%
- ✅ **Explainability**: Bisa menjelaskan alasan keputusan

---

## 📧 Kontak

**Peneliti**: Brian Rizqi  
**Institusi**: [Universitas Anda]  
**Email**: [Email Anda]

---

## 📝 Catatan Penting

1. **Pertama kali run akan lama** karena download model AI (~2-5GB)
2. **Butuh koneksi internet** untuk download model
3. **CPU mode lebih lambat** tapi lebih stabil (tidak perlu GPU)
4. **Hasil disimpan** di file `local_demo_result_cpu.json`

---

## 🎓 Untuk Publikasi

Project ini adalah bagian dari penelitian postdoc dengan target publikasi di jurnal internasional Q1/Q2 seperti:
- Expert Systems with Applications
- Information Processing & Management
- Digital Journalism

**Kontribusi Penelitian:**
- Framework Agentic AI untuk deteksi native ads
- Dataset native ads Indonesia (akan dipublikasikan)
- Sistem dengan explainability tinggi
