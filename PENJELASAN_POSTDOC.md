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

**Apa itu Agentic AI?** (Bahasa Bayi Version)

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

---

## 🗣️ Jawaban untuk Pertanyaan Promotor

### Q1: "Jadi kamu pakai AI apa?"
**A**: Saya pakai **Agentic AI**, yaitu sistem AI yang terdiri dari **banyak agen** yang bekerja sama. Setiap agen punya tugas spesifik (scraping, preprocessing, retrieval, classification, explanation).

### Q2: "Bedanya dengan machine learning biasa?"
**A**: 
- **ML Biasa**: 1 model, input → output, tidak bisa jelaskan alasan
- **Agentic AI**: Banyak agen, setiap agen punya tugas, bisa jelaskan alasan keputusan

### Q3: "Kenapa native ads? Kenapa tidak spam email?"
**A**: 
- Native ads adalah **masalah baru** di era digital
- Belum banyak penelitian (gap penelitian)
- Punya **dampak sosial** (masyarakat tertipu)
- Relevan dengan **media transparency**

### Q4: "Datanya dari mana?"
**A**: 
- Scraping dari portal berita Indonesia (Detik, Kompas, Tribun, CNN Indonesia, Tempo)
- Total: **12,088 artikel** (6,044 berita asli + 6,044 native ads)
- Sudah dilabel manual oleh expert

### Q5: "Akurasinya berapa?"
**A**: 
- Target: **>90%**
- Baseline (ML biasa): 70-80%
- Improvement: **+10-20%**

### Q6: "Noveltynya apa?"
**A**: 
1. **Pertama** yang pakai Agentic AI untuk native ads detection
2. **Pertama** yang bikin dataset native ads Indonesia
3. **Pertama** yang fokus ke explainability (bisa jelaskan keputusan)

### Q7: "Aplikasinya apa?"
**A**: 
- Portal berita bisa pakai untuk **auto-label** artikel
- Regulator bisa pakai untuk **monitoring** transparansi media
- Peneliti lain bisa pakai **dataset** untuk riset

### Q8: "Berapa lama penelitiannya?"
**A**: 
- **Fase 1** (Bulan 1-2): Bikin sistem
- **Fase 2** (Bulan 2-3): Kumpulkan data
- **Fase 3** (Bulan 3-4): Eksperimen
- **Fase 4** (Bulan 4-5): Evaluasi
- **Fase 5** (Bulan 5-6): Tulis paper & submit

---

## 🎯 Kesimpulan (TL;DR)

**Penelitian ini tentang:**
- Bikin sistem AI untuk deteksi iklan menyamar di berita online
- Pakai teknologi Agentic AI (banyak agen bekerja sama)
- Akurasi tinggi (>90%) dan bisa jelaskan alasan
- Kontribusi: Framework baru + Dataset Indonesia + Open Source

**Kenapa penting:**
- Masalah baru di era digital
- Belum ada yang pakai Agentic AI untuk ini
- Punya dampak sosial (transparansi media)

**Target:**
- Publikasi di jurnal Q1/Q2
- Dataset dipublikasikan untuk peneliti lain
- Sistem bisa dipakai portal berita Indonesia

---

## 📚 Referensi Utama (Kalau Promotor Minta)

1. **Xi et al. (2023)** - "The Rise and Potential of LLM-based Agents"
2. **Wang et al. (2024)** - "A Survey on Large Language Model based Agents"
3. **Wojdynski & Evans (2016)** - "Going Native: Effects of Disclosure Position and Language on Recognition of Online Native Advertising"
4. **Lewis et al. (2020)** - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

---

**Catatan**: Kalau promotor masih bingung, ajak lihat **demo langsung**! Lebih mudah paham kalau lihat sistem jalan 😊
