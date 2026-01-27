# Evaluation Report: Native Ads vs. Berita Murni Classification

## Executive Summary
The evaluation of **v6** models shows that **Qwen (Optimized)** and **Llama** are the top performers with accuracies around **70%**. However, they exhibit significant biases: Llama and Qwen favor "Native Ads" (high sensitivity, low specificity), while Gemma favors "Berita Murni" (pure news).

| Model | Overall Accuracy | Native Ads Acc | Berita Murni Acc | Primary Bias |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen v6 (Optimized)** | **70.50%** | 93.75% | 49.04% | Native Ads |
| **Llama v6** | 69.00% | **98.00%** | 40.00% | Native Ads |
| **Gemma v6** | 66.50% | 39.00% | **94.00%** | Berita Murni |
| **GPT-OSS v6** | 46.00% | 65.00% | 27.00% | Inconsistent |

---

## Key Findings

### 1. Model Specific Performance
- **Qwen v6 (Optimized)**: Currently the best overall model at **70.5%**. The transition to the `optimized` dataset provided a massive boost to its native ads detection (from 12% to 93.75%).
- **Llama v6**: Nearly perfect at identifying native ads (98%) but tends to hallucinate "promotion" in regular news, especially if a brand or entity is mentioned (e.g., classifying a news piece about Rafael Nadal as a native ad).
- **Gemma v6**: Highly reliable for identifying "Berita Murni" (94%) but struggles to detect subtle native ads.

### 2. Dataset Impact
Comparison of Qwen v6 on different datasets reveals that the `llm_dataset_finetuning_optimized.json` significantly improved performance:
- **Clean Dataset**: 53.5% Accuracy
- **Optimized Dataset**: 70.5% Accuracy (+17%)

### 3. Common Error Patterns
- **Entity Confusion**: News reporting on specific brands (e.g., "PO ALS bus fire", "Garuda Indonesia performance") are frequently misclassified as Native Ads by Llama and Qwen because they mention positive traits or specific brand names.
- **Tone Misinterpretation**: Positive news about government programs or social events are often mistaken for persuasive advertising.
- **Output Stability**: Qwen frequently produces "Incomplete JSON" or "No JSON found", which may affect parsing reliability in production.

---

## Recommendations

### Short-term
- **Use Qwen v6 (Optimized)** as the base model for further refinement due to its balanced (yet still biased) performance.
- **Ensemble Approach**: Combining **Gemma** (good at Berita Murni) and **Llama** (good at Native Ads) via a voting mechanism could potentially yield >80% accuracy.

### Long-term (Fine-Tuning)
- **Hard Negative Mining**: Incorporate more "Pure News" samples that mention brands but are non-promotional (sports, financial reports, accidents) into the fine-tuning set to reduce False Positives.
- **Prompt Engineering**: Refine the prompt to explicitly instruct models on how to distinguish between "Corporate Social Responsibility/News" and "Persuasive Native Ads".
- **Sequence Length**: Ensure the training/inference window is long enough to capture the full context, as subtle ads often hide their intent in the latter half of the text.
