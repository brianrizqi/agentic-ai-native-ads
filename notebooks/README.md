# Jupyter Notebooks - Native Ads Detection

This folder contains interactive Jupyter notebooks for the Native Ads Detection system.

## 📚 Notebooks Overview

### [00_dataset_processing.ipynb](./00_dataset_processing.ipynb)
**Dataset Collection & Processing**
- Convert Excel dataset to JSON format
- LLM augmentation for detailed reasoning
- Dataset merging and balancing
- Format conversion for fine-tuning

**Use Case:** Prepare dataset for model training

---

### [01_classification_demo.ipynb](./01_classification_demo.ipynb)
**Classification Demo**
- Single URL classification
- Batch processing
- Result visualization
- Custom URL testing

**Use Case:** Demo and testing the classification system

---

### [02_finetuning.ipynb](./02_finetuning.ipynb)
**Model Fine-Tuning**
- Load base model with Unsloth
- Add LoRA adapters
- Train on custom dataset
- Save fine-tuned model

**Use Case:** Train custom model for native ads detection

**⚠️ Requirements:** GPU with CUDA support

---

### [03_model_evaluation.ipynb](./03_model_evaluation.ipynb)
**Model Evaluation**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization
- Error analysis
- Per-class metrics

**Use Case:** Evaluate model performance

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install -r ../requirements.txt

# Fine-tuning dependencies (for notebook 02)
pip install -r ../langchain-refactor/requirements-finetuning.txt

# Evaluation dependencies (for notebook 03)
pip install -r ../langchain-refactor/requirements-evaluation.txt

# Jupyter
pip install jupyter notebook
```

### 2. Launch Jupyter

```bash
cd notebooks
jupyter notebook
```

### 3. Run Notebooks

Open any notebook and run cells sequentially (Shift+Enter)

---

## 📋 Workflow

**Complete Pipeline:**

```
00_dataset_processing.ipynb
    ↓ (prepare dataset)
02_finetuning.ipynb
    ↓ (train model)
03_model_evaluation.ipynb
    ↓ (evaluate performance)
01_classification_demo.ipynb
    ↓ (use for classification)
```

**Quick Demo:**

```
01_classification_demo.ipynb
    (use pre-trained model)
```

---

## 💡 Tips

### For Research/Paper
- Use **03_model_evaluation.ipynb** to generate metrics and confusion matrix
- Export results to `../results/` for inclusion in paper

### For Demo/Presentation
- Use **01_classification_demo.ipynb** for live demonstrations
- Show real-time classification with custom URLs

### For Development
- Use **00_dataset_processing.ipynb** to experiment with dataset variations
- Use **02_finetuning.ipynb** to test different hyperparameters

---

## 📁 Output Locations

- **Results:** `../results/`
- **Models:** `../models/`
- **Datasets:** `../data/`

---

## 🔧 Troubleshooting

### GPU Memory Issues (Fine-tuning)
```python
# In notebook 02, reduce batch size
BATCH_SIZE = 1  # Already minimal
# Or reduce max_seq_length
max_seq_length = 1024  # Instead of 2048
```

### Slow Evaluation
```python
# In notebook 03, reduce sample size
NUM_SAMPLES = 50  # Instead of 200
```

### Import Errors
```bash
# Make sure you're in the notebooks/ directory
cd notebooks
# And paths are correct
sys.path.append('../langchain-refactor')
```

---

## 📊 Expected Results

### Classification (Notebook 01)
- **Inference Time:** ~10-15 seconds per URL
- **Accuracy:** ~85-90% (with fine-tuned model)

### Fine-tuning (Notebook 02)
- **Training Time:** ~2-4 hours (14B model, 2000 steps)
- **GPU Memory:** ~24GB VRAM required

### Evaluation (Notebook 03)
- **Metrics:** Accuracy, Precision, Recall, F1
- **Visualizations:** Confusion matrix, error analysis

---

## 🎯 Use Cases

| Notebook | Research | Demo | Development |
|----------|----------|------|-------------|
| 00 | ✅ Dataset prep | ❌ | ✅ Experimentation |
| 01 | ✅ Results | ✅ Live demo | ✅ Testing |
| 02 | ✅ Model training | ❌ | ✅ Hyperparameter tuning |
| 03 | ✅ Metrics for paper | ✅ Performance viz | ✅ Model comparison |

---

## 📝 Notes

- All notebooks are self-contained with explanations
- Results are automatically saved to `../results/`
- Notebooks can be run independently (except 02 requires 00)
- GPU recommended for notebooks 02 and 03

---

## 🤝 Contributing

To add a new notebook:
1. Follow naming convention: `XX_descriptive_name.ipynb`
2. Include markdown explanations
3. Add to this README
4. Test on clean environment

---

## 📚 References

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [LangChain Documentation](https://python.langchain.com/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
