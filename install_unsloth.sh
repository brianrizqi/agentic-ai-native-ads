#!/bin/bash
# Install Unsloth and dependencies for fine-tuning

echo "================================================================================"
echo "Installing Unsloth for Fine-Tuning"
echo "================================================================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Upgrade torchvision first
echo "📦 Upgrading torchvision for compatibility..."
pip install --upgrade torchvision

# Install Unsloth
echo ""
echo "📦 Installing Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install additional dependencies
echo ""
echo "📦 Installing additional dependencies..."
pip install --no-deps trl peft accelerate bitsandbytes

# Verify installation
echo ""
echo "✅ Verifying installation..."
python3 -c "from unsloth import FastLanguageModel; print('Unsloth installed successfully!')"

echo ""
echo "================================================================================"
echo "Installation Complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "1. cd langchain-refactor"
echo "2. ./train_all_models.sh"
echo ""
