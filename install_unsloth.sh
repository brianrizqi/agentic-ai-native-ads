#!/bin/bash
# Install Unsloth and dependencies for fine-tuning
# Usage: bash install_unsloth.sh

echo "Installing Unsloth for fine-tuning..."

# Install Unsloth from git
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install training dependencies without deps to avoid conflicts
pip install --no-deps trl peft accelerate bitsandbytes

# Install remaining dependencies
pip install -r requirements-unsloth.txt

echo "✅ Unsloth installation complete!"
echo ""
echo "Test with:"
echo "python finetune_unsloth.py --help"
