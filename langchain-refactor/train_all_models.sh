#!/bin/bash
# Train all 4 models sequentially for native ads detection
# Run from langchain-refactor directory

set -e  # Exit on error

echo "================================================================================"
echo "MULTI-MODEL TRAINING PIPELINE"
echo "Training 4 models: Qwen 14B, GPT-OSS 20B, Llama 3.1 8B, Gemma 2 9B"
echo "================================================================================"
echo ""

# Configuration
DATASET="../data/llm_dataset_mixed_json.json"
MAX_STEPS=1500
RESULTS_DIR="../results/multi_model_training"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/training_log_$(date +%Y%m%d_%H%M%S).txt"

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Max Steps: $MAX_STEPS"
echo "  Results: $RESULTS_DIR"
echo "  Log: $LOG_FILE"
echo ""

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
    echo "❌ Dataset not found: $DATASET"
    echo "Please run dataset preparation first!"
    exit 1
fi

# Function to train a model
train_model() {
    local model=$1
    local start_time=$(date +%s)
    
    echo "================================================================================"
    echo "Training: $model"
    echo "Start time: $(date)"
    echo "================================================================================"
    
    python finetune.py \
        --model "$model" \
        --dataset "$DATASET" \
        --max-steps "$MAX_STEPS" \
        2>&1 | tee -a "$LOG_FILE"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    
    echo ""
    echo "✅ $model training complete!"
    echo "   Duration: ${hours}h ${minutes}m"
    echo ""
    
    # Save timing info
    echo "$model: ${hours}h ${minutes}m" >> "$RESULTS_DIR/training_times.txt"
}

# Start training
echo "🚀 Starting multi-model training pipeline..."
echo "⚠️  This will take approximately 8-16 hours"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Train all models
echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "TRAINING PIPELINE STARTED: $(date)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Model 1: Llama 3.1 8B (fastest, good baseline)
train_model "llama"

# Model 2: Gemma 2 9B
train_model "gemma"

# Model 3: Qwen 2.5 14B (best multilingual)
train_model "qwen"

# Model 4: GPT-OSS 20B (largest, experimental - uses dedicated script)
echo "================================================================================"
echo "Training: gpt-oss (using dedicated script)"
echo "Start time: $(date)"
echo "================================================================================"

python finetune_gpt_oss.py \
    --dataset "$DATASET" \
    --max-steps "$MAX_STEPS" \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "✅ gpt-oss training attempt complete!"
echo ""

# Summary
echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "ALL MODELS TRAINED SUCCESSFULLY!" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Training Times:" | tee -a "$LOG_FILE"
cat "$RESULTS_DIR/training_times.txt" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Next steps:" | tee -a "$LOG_FILE"
echo "1. Run comparison: python compare_models.py" | tee -a "$LOG_FILE"
echo "2. View results: cat $RESULTS_DIR/model_comparison.json" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Models saved in:" | tee -a "$LOG_FILE"
echo "  - ../models/llama-native-ads_merged_16bit" | tee -a "$LOG_FILE"
echo "  - ../models/gemma-native-ads_merged_16bit" | tee -a "$LOG_FILE"
echo "  - ../models/qwen-native-ads_merged_16bit" | tee -a "$LOG_FILE"
echo "  - ../models/gpt-oss-native-ads_merged_16bit" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
