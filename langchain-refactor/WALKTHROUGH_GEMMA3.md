# Walkthrough - Improving Gemma 3 Performance & LangChain LoRA Support

Saya telah memperbaiki masalah performa rendah pada Gemma 3 270M dengan menyamakan format prompt antara fine-tuning dan evaluasi, serta menambahkan dukungan LoRA adapters ke sistem LangChain.

## Perubahan Utama

### 1. Prompt Alignment untuk Gemma 3
Model Gemma 3 (terutama versi 270M) sangat sensitif terhadap format prompt. Saya telah mengubah `LLMClassifierAgent` (legacy) dan `ClassificationAgent` (LangChain) untuk menggunakan format yang sama dengan saat fine-tuning:
- Menghapus tag `[INST]` (khusus Mistral) saat mendeteksi model Gemma 3.
- Menggunakan instruksi minimalis yang fokus pada klasifikasi JSON.
- Memperbaiki typo "INFORMITIF" menjadi "INFORMATIF" agar sesuai dengan label yang dipelajari model.

### 2. Dukungan LoRA di LangChain Agent
Saya telah menambahkan parameter `lora_path` pada `ClassificationAgent` di direktori `langchain-refactor`. Sekarang sistem bisa memuat adapter PEFT secara otomatis saat menggunakan provider `local`.

### 3. Update Skrip Evaluasi
Skrip evaluasi di `langchain-refactor` sekarang mendukung argumen `--lora-path`.

## Cara Menjalankan Evaluasi (di Komputer Target)

Karena instruksi Anda untuk tidak menjalankan kode di komputer ini, silakan jalankan perintah berikut di komputer target Anda:

### Evaluasi Full (LangChain)
```bash
cd langchain-refactor
python3 evaluate_model.py \
    --model google/gemma-3-270m \
    --lora-path /path/to/your/gemma3-lora \
    --num-samples 100
```

### Evaluasi Khusus Gemma 3 (Hugging Face)
```bash
cd langchain-refactor
python3 evaluate_gemma3_hf.py \
    --base_model google/gemma-3-270m \
    --model_path /path/to/your/gemma3-lora \
    --num-samples 100
```

### Menjalankan Demo LangChain
```bash
cd langchain-refactor
python3 main.py \
    --url "https://contoh-berita.com" \
    --provider local \
    --model google/gemma-3-270m \
    --lora-path /path/to/your/gemma3-lora
```

## Hasil yang Diharapkan
Dengan prompt yang selaras, Recall untuk **Native Ads** seharusnya meningkat signifikan, dan model tidak lagi bingung dengan instruksi yang terlalu panjang atau format yang salah.
