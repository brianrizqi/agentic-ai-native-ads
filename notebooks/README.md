# Notebooks - Agentic AI Native Ads Detection

Folder ini berisi Jupyter notebooks untuk penelitian postdoc.

## 📚 Notebooks

### 1. `01_dataset_conversion.ipynb`
**Konversi Dataset Excel → LLM Formats**

- Input: `native_ads_dataset.xlsx` (12,088 samples)
- Output: 
  - `llm_dataset_qna.json` - untuk RAG
  - `llm_dataset_instruction.json` - untuk fine-tuning
  - `llm_dataset_chat.jsonl` - untuk OpenAI-style

**Cara pakai**:
```bash
jupyter notebook notebooks/01_dataset_conversion.ipynb
```

### 2. `02_finetuning_colab.ipynb`
**Fine-Tuning di Google Colab (RECOMMENDED)**

- Platform: Google Colab (gratis, GPU T4/A100)
- Model: Mistral-7B-Instruct-v0.2
- Method: LoRA (efisien)
- Duration: ~2-3 jam

**Cara pakai**:
1. Buka di Google Colab: [Open in Colab](https://colab.research.google.com)
2. Upload notebook `02_finetuning_colab.ipynb`
3. Runtime → Change runtime type → GPU (T4 atau A100)
4. Run all cells
5. Download model hasil training

**Keuntungan Colab**:
- ✅ GPU gratis (T4 16GB atau A100 40GB)
- ✅ Tidak ada MIG issues
- ✅ Setup mudah
- ✅ Bisa pause/resume

## 🚀 Quick Start

### Option A: Local (Jika punya GPU tanpa MIG)

```bash
# 1. Konversi dataset
jupyter notebook notebooks/01_dataset_conversion.ipynb

# 2. Fine-tune (local)
python finetune_model.py --use-lora
```

### Option B: Google Colab (RECOMMENDED)

```bash
# 1. Konversi dataset (local)
jupyter notebook notebooks/01_dataset_conversion.ipynb

# 2. Upload llm_dataset_instruction.json ke Colab
# 3. Run 02_finetuning_colab.ipynb di Colab
# 4. Download model hasil training
```

## 📊 Expected Results

Setelah fine-tuning:
- **Accuracy**: 85-95%
- **BERTScore F1**: 0.85-0.92
- **LLM Judge Overall**: 4.0-4.5/5.0

## 💡 Tips

1. **Untuk testing cepat**: Gunakan subset data (1000 samples)
2. **Untuk publikasi**: Gunakan full dataset (12,088 samples)
3. **Jika error MIG**: Gunakan Google Colab
4. **Save checkpoints**: Training bisa di-resume jika terputus

## 📞 Support

Lihat dokumentasi lengkap:
- `COMPLETE_GUIDE.md` - Panduan lengkap
- `FINETUNING_GUIDE.md` - Detail fine-tuning
- `EVALUATION_GUIDE.md` - Evaluasi model
