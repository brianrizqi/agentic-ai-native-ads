# Penjelasan Postdoc

## 🎯 Masalahnya Apa?

Bayangkan Anda baca berita di internet. Ada 2 jenis artikel:

1. **Berita Asli** 📰
   - Contoh: "Harga BBM Naik Rp500 Mulai Besok"
   - Tujuan: Kasih informasi ke pembaca

2. **Iklan Menyamar (Native Ads)** 💰
   - Contoh: "Tips Hemat BBM dengan Oli Merek X - Terbukti Irit!"
   - Tujuan: Jualan produk, tapi ditulis seperti berita

**Masalahnya**: Pembaca sering **tidak sadar** kalau sedang baca iklan! Ini bisa **menyesatkan**.

---

## 🤔 Kenapa Ini Penting?

### Untuk Masyarakat:
- Orang bisa **tertipu** beli produk yang tidak perlu
- Kepercayaan ke media jadi **turun**
- Tidak ada **transparansi**

### Untuk Akademik:
- Belum ada sistem **otomatis** yang akurat untuk deteksi native ads
- Metode lama (rule-based, machine learning biasa) kurang **akurat**
- Belum ada yang pakai **Agentic AI** untuk kasus ini

---

## 💡 Solusi yang Saya Tawarkan

Saya buat sistem AI yang bisa **otomatis membedakan** berita asli vs iklan menyamar.

### Teknologi yang Dipakai: **Agentic AI**

**Apa itu Agentic AI?**

Bayangkan Anda punya **tim robot** yang bekerja sama:

```
Robot 1 (Web Scraper)    → Ambil artikel dari internet
         ↓
Robot 2 (Preprocessor)   → Bersihkan teks (hapus iklan banner, dll)
         ↓
Robot 3 (Retriever)      → Cari contoh kasus serupa di database
         ↓
Robot 4 (Classifier)     → Putuskan: Ini berita atau iklan?
         ↓
Robot 5 (Explainer)      → Jelaskan alasannya dengan bahasa manusia
```

**Kenapa pakai banyak robot?**
- Kalau cuma 1 robot → kurang akurat
- Kalau pakai tim robot → setiap robot punya tugas spesifik → lebih akurat!

---

## 🤖 Model AI yang Dipakai

### ⚠️ Penting: Tidak Semua Agen Pakai AI!

Dari 5 agen, hanya **3 agen yang pakai model AI**. Yang lain pakai library programming biasa.

### Agen yang TIDAK Pakai AI:

#### 1. **Web Scraper Agent** (Tidak Pakai AI)
**Library**: BeautifulSoup4 + Requests  
**Fungsi**: Ambil artikel dari website (seperti copy-paste otomatis)

**Analogi**: Seperti **fotokopi**
- Anda pergi ke perpustakaan
- Fotokopi halaman buku yang Anda butuhkan
- Tidak perlu AI, cuma butuh mesin fotokopi

**Cara Kerja**:
1. Buka URL berita
2. Download HTML-nya
3. Extract teks artikel (buang iklan, sidebar, footer)

#### 2. **Preprocessing Agent** (Tidak Pakai AI)
**Library**: NLTK, spaCy, Regex  
**Fungsi**: Bersihkan teks sebelum dianalisis AI

**Analogi**: Seperti **cuci sayuran** sebelum dimasak
- Buang daun yang layu (emoji, special characters)
- Potong-potong (tokenisasi)
- Buang bagian yang tidak perlu (stopwords)

**Cara Kerja**:
```
Input: "Wah!!! Promo GEDE-GEDEAN 🎉 diskon 50% lho!!!"
         ↓ (cleaning)
Output: "promo besar diskon 50 persen"
```

---

### Agen yang PAKAI AI (3 Agen):

#### 3. **Retriever Agent** ✅ PAKAI AI
**Nama**: `sentence-transformers/all-MiniLM-L6-v2`  
**Ukuran**: 80MB (kecil!)  
**Fungsi**: Ubah kalimat jadi angka-angka (vector) untuk bisa dicari

**Analogi**: Seperti **barcode** di supermarket
- Setiap produk punya barcode unik
- Kasir scan barcode → langsung tahu produknya apa
- Model ini ubah artikel jadi "barcode" → bisa cari artikel serupa

#### 4. **LLM Classifier Agent** (Otak Utama) ✅ PAKAI AI
**Nama**: `mistralai/Mistral-7B-Instruct-v0.2`  
**Ukuran**: 7GB  
**Fungsi**: Baca artikel → Putuskan: Native Ads atau Editorial?

**Kenapa Mistral?**
- ✅ Open source (gratis)
- ✅ Akurasi tinggi (90%+)
- ✅ Support bahasa Indonesia
- ✅ Bisa jalan di GPU biasa

**Alternatif (kalau tidak ada GPU):**
- `microsoft/phi-2` (2.7GB) → Lebih kecil, bisa jalan di CPU
- `Qwen2.5-3B` (3GB) → Alternatif lain untuk CPU

#### 5. **Explanation Agent** ✅ PAKAI AI
**Nama**: `mistralai/Mistral-7B-Instruct-v0.2` (sama dengan Classifier)  
**Fungsi**: Jelaskan alasan keputusan dengan bahasa manusia

**Contoh Output**:
```
Artikel ini diklasifikasikan sebagai NATIVE ADS karena:
1. Ada kata promosi: "diskon", "promo spesial"
2. Ada call-to-action: "buruan daftar"
3. Menyebut brand 15 kali
4. Gaya bahasa marketing, bukan jurnalistik
```

### 📊 Perbandingan Model (Sederhana):

| Model | Ukuran | Butuh GPU? | Akurasi | Kecepatan |
|-------|--------|------------|---------|----------|
| **Mistral-7B** | 7GB | Ya (recommended) | 90%+ | Sedang |
| **Phi-2** | 2.7GB | Tidak (CPU OK) | 85%+ | Cepat |
| **GPT-OSS-20B** | 20GB | Ya (GPU kuat) | 95%+ | Lambat |
| **Qwen2.5-3B** | 3GB | Tidak (CPU OK) | 87%+ | Cepat |

**Pilihan Terbaik**:
- Punya GPU → Pakai **Mistral-7B** (akurasi tinggi)
- Tidak punya GPU → Pakai **Phi-2** (cepat di CPU)
- Untuk paper (butuh akurasi maksimal) → Pakai **GPT-OSS-20B**

---

## 🔬 Cara Kerjanya (Analogi Sederhana)

### Analogi 1: Tim Dokter 👨‍⚕️

Seperti pasien yang diperiksa tim dokter:

1. **Perawat** → Catat data pasien (nama, umur, keluhan)
2. **Dokter Umum** → Periksa fisik
3. **Dokter Spesialis** → Cek riwayat penyakit serupa
4. **Dokter Ahli** → Diagnosis: Flu atau COVID?
5. **Dokter Konsultan** → Jelaskan ke pasien: "Ini flu biasa karena X, Y, Z"

### Analogi 2: Tim Detektif 🕵️

Seperti polisi yang selidiki kasus:

1. **Detektif 1** → Kumpulkan bukti dari TKP
2. **Detektif 2** → Analisis bukti (sidik jari, DNA)
3. **Detektif 3** → Cari kasus serupa di database
4. **Detektif 4** → Tentukan tersangka
5. **Detektif 5** → Buat laporan untuk jaksa

---

## 📊 Keunggulan Sistem Saya

### 1. **Akurat** ✅
- Target akurasi: >90%
- Lebih tinggi dari metode lama (70-80%)

### 2. **Bisa Menjelaskan** 🗣️
- Tidak cuma bilang "Ini iklan"
- Tapi juga kasih alasan: "Ini iklan karena ada kata promosi, diskon, call-to-action"

### 3. **Otomatis** 🤖
- Tidak perlu manusia cek satu-satu
- Bisa proses ribuan artikel per hari

### 4. **Belajar Terus** 📚
- Kalau ada kasus baru → sistem bisa belajar
- Makin lama makin pintar

---

## 🎓 Kontribusi Penelitian (Untuk Akademik)

### Kontribusi Teori:
1. **Framework baru**: Agentic AI untuk native ads detection (belum ada yang buat)
2. **Multi-agent architecture**: Pakai 5 agen yang bekerja sama
3. **Explainability**: Sistem bisa jelaskan keputusan (penting untuk trust)

### Kontribusi Praktis:
1. **Dataset Indonesia**: Kumpulan 12,000+ artikel berita Indonesia (akan dipublikasikan)
2. **Open Source**: Kode akan dibuka untuk peneliti lain
3. **Real-world impact**: Bisa dipakai portal berita untuk transparansi

---

## 📈 Target Publikasi

### Jurnal Internasional (Q1/Q2):
- **Expert Systems with Applications** (IF: 8.5)
- **Information Processing & Management** (IF: 7.4)
- **Digital Journalism** (IF: 4.2)

### Konferensi:
- AAAI (AI Conference)
- ACL (Natural Language Processing)
- WWW (Web Conference)