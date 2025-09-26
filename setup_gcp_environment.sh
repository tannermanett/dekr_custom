#!/bin/bash

# Environment setup script for Google Cloud VM
# Run this ON the GCP VM after cloning the repository

echo "🔧 Setting up DEKR training environment on Google Cloud..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install additional dependencies
echo "📦 Installing additional dependencies..."
sudo apt install -y git wget unzip htop tmux

# Verify GPU is available
echo "🎮 Checking GPU availability..."
nvidia-smi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install additional packages for better performance
echo "📦 Installing performance packages..."
pip install tensorboard wandb

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/coco
mkdir -p output
mkdir -p log

# Set up environment variables for optimal performance
echo "⚙️  Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'

# DEKR Training Environment Variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# TensorBoard
export TENSORBOARD_PORT=6006

EOF

# Reload bashrc
source ~/.bashrc

echo "✅ Environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Upload your dataset to data/coco/"
echo "2. Run training: ./gcp_optimized_train.sh"
echo ""
echo "📊 Monitor training:"
echo "   - GPU usage: nvidia-smi"
echo "   - System resources: htop"
echo "   - Training logs: tail -f training.log"
echo ""
echo "🌐 Access TensorBoard (if enabled):"
echo "   - From local machine: gcloud compute ssh $VM_NAME --zone=$ZONE -- -L 6006:localhost:6006"
echo "   - Then open: http://localhost:6006"
echo ""
echo "💰 Remember to stop the VM when not training to save money!"
