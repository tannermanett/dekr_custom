#!/bin/bash

# Script to fix shared memory issues for PyTorch training
# Run this before training to increase shared memory limits

echo "Current shared memory status:"
df -h /dev/shm

echo ""
echo "Setting environment variables to reduce shared memory usage..."

# Export environment variables to reduce shared memory pressure
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

echo "Environment variables set:"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"

echo ""
echo "You can now run your training script."
echo "Recommended: ./conservative_train.sh"
