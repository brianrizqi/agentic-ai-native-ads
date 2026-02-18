#!/bash
# setup_gemma3.sh - Setup for Gemma 3 Fine-tuning

echo "🚀 Setting up environment for Gemma 3 Fine-tuning..."

# 1. Install Unsloth and core dependencies
# Unsloth will handle the correct versions of transformers/torch internally
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# 2. Login to Hugging Face
echo "🔑 Please provide your Hugging Face Token when prompted below."
echo "You can get one at: https://huggingface.co/settings/tokens"

# Use python -m to avoid 'command not found' if PATH is not set
python3 -m huggingface_hub.commands.user login

echo "✅ Setup complete!"
echo "To run training:"
echo "export HF_TOKEN=your_token_here (if not logged in via CLI)"
echo "python3 langchain-refactor/finetune.py --model gemma3"
