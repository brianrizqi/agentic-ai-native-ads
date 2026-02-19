# Panduan Fine-tuning (Gemma 3 & Model Lainnya)

Jika Anda ingin melakukan fine-tuning ulang untuk meningkatkan performa, berikut adalah langkah-langkah praktisnya.

## 1. Persiapan Lingkungan
Pastikan Anda sudah login ke Hugging Face karena model Gemma 3 berada di repositori yang diproteksi.

```bash
# Login via CLI
huggingface-cli login

# Atau set environment variable
export HF_TOKEN=your_token_here
```

## 2. Persiapan Data (Crucial)
Pastikan dataset Anda sudah melalui tahap optimasi (ekstrak judul dan potong reasoning yang terlalu panjang).

```bash
cd langchain-refactor
python3 tools/prepare_finetuning_dataset.py \
  --input ../data/llm_dataset_12k_refined.json \
  --output ../data/llm_dataset_finetuning_ready.json
```

## 2. Proses Fine-tuning
Gunakan script `finetune.py` yang sudah dioptimasi. Script ini menggunakan **Unsloth** untuk training yang jauh lebih cepat.

### Opsi A: Gemma 3 270M (Super Ringan)
```bash
python3 finetune.py \
  --model gemma3 \
  --dataset ../data/llm_dataset_finetuning_ready.json \
  --output ../models/gemma3-finetuned \
  --max-steps 1500
```

### Opsi B: Llama 3.1 8B (Sangat Direkomendasikan jika GPU Cukup)
Jika Anda memiliki GPU dengan VRAM > 16GB, Llama 8B akan memberikan hasil jauh lebih cerdas dibanding Gemma 270M.
```bash
python3 finetune.py \
  --model llama \
  --dataset ../data/llm_dataset_finetuning_ready.json \
  --output ../models/llama-finetuned \
  --max-steps 1500
```

## 3. Langkah Setelah Training
Training akan menghasilkan dua versi model:
1. `models/your-model/`: Ini adalah LoRA adapters (kecil).
2. `models/your-model_merged_16bit/`: Ini adalah model utuh yang sudah digabung.

**Gunakan model versi merged untuk evaluasi dan produksi.**

## 4. Evaluasi Hasil
Setelah selesai, langsung cek akurasinya:
```bash
python3 evaluate_model.py \
  --model ../models/gemma3-finetuned_merged_16bit \
  --num-samples 100
```

---
> [!TIP]
> **Mengapa Perlu Re-finetune?**
> Jika Anda mengubah template prompt atau menambahkan data baru, model perlu dilatih ulang agar "terbiasa" dengan pola baru tersebut. Prompt yang saya rekomendasikan sekarang sudah saya masukkan secara default di `finetune.py`.
