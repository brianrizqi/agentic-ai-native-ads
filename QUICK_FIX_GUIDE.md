# Quick Fix Guide - Qwen Model Bias

## Problem
Model memprediksi SEMUA sampel sebagai "native ads" → Accuracy 47.8%

## Root Cause
✅ Dataset BALANCED (50:50) - bukan masalah data
❌ Prompt BIASED (2:1 native ads examples) - **INI MASALAHNYA!**

## Solution Applied

### 1. Improved Prompt (DONE ✅)
File: `agents/llm_classifier_agent.py`

**Changes:**
- Added balanced examples: 3 native ads + 3 berita murni (was 2:1)
- Added explicit instructions: "NOT everything is native ads!"
- Added guidance: "When in doubt, prefer berita murni"
- Improved example quality with clearer distinctions

### 2. Test the Fix

#### Option A: Quick Test (1 sample)
```bash
python run_local_demo.py \
  --model models/native-ads-qwen25-lora \
  --text "Presiden Jokowi menghadiri pertemuan G20 di Bali hari ini."
```

Expected: Should predict "berita murni" (not native ads!)

#### Option B: Full Re-evaluation (500 samples)
```bash
python comprehensive_evaluation_full.py \
  --model models/native-ads-qwen25-lora \
  --dataset data/llm_dataset_instruction.json \
  --num-samples 500 \
  --output data/eval_qwen_fixed.json
```

Expected improvements:
- Accuracy: 47.8% → **70-80%**
- Berita Murni Recall: 0% → **60-75%**
- Balanced predictions (not all native ads)

### 3. Compare Results

```bash
# Before (old results)
cat data/eval_qwen_full.json | grep -A 10 '"traditional"'

# After (new results)
cat data/eval_qwen_fixed.json | grep -A 10 '"traditional"'
```

## If Still Not Working

### Plan B: Retrain Model with Improvements

Create `finetune_model_improved.py` with:
1. **Label smoothing** (0.1) - prevent overconfidence
2. **Lower learning rate** (1e-5 instead of 2e-5)
3. **More epochs** (5 instead of 3)
4. **Gradient clipping** (max_grad_norm=1.0)

```bash
python finetune_model_improved.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset data/llm_dataset_instruction.json \
  --output models/native-ads-qwen25-v2 \
  --epochs 5 \
  --batch-size 4 \
  --use-lora
```

### Plan C: Try Different Base Model

Models to try:
- `meta-llama/Llama-3.2-3B-Instruct` (better instruction following)
- `mistralai/Mistral-7B-Instruct-v0.3` (more balanced)

## Expected Timeline

- **Immediate** (5 min): Test with improved prompt
- **Short-term** (30 min): Full re-evaluation
- **If needed** (2-3 hours): Retrain model

## Success Criteria

✅ Berita Murni Recall > 60%
✅ Overall Accuracy > 70%
✅ Balanced confusion matrix (not all native ads)
✅ F1-Score > 0.65 for both classes
