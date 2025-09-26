#!/bin/bash

# Google Cloud Platform optimized training script for DEKR
# Designed for T4 GPU with higher memory and better performance

cd /home/wner/DEKR

# Checkpoint path
CKPT=output/coco_kpt/hrnet_dekr/w32_rink_512/checkpoint.pth.tar

# GCP-optimized training command
# T4 GPU has 16GB VRAM, so we can use larger batch sizes
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
python tools/train.py \
  --cfg experiments/rink/w32/w32_rink_512.yaml \
  TRAIN.RESUME True \
  TRAIN.CHECKPOINT $CKPT \
  TRAIN.IMAGES_PER_GPU 16 \
  WORKERS 8 \
  AMP True \
  TF32 True \
  CHANNELS_LAST True \
  PIN_MEMORY True \
  PREFETCH_FACTOR 4 \
  PERSISTENT_WORKERS True \
  FILE_SYSTEM_SHARING True \
  PRINT_FREQ 10

# GCP T4 optimizations:
# 1. Batch size 16 (T4 has 16GB VRAM vs local RTX 4060's 8GB)
# 2. 8 workers (GCP VMs have more CPU cores)
# 3. All memory optimizations enabled (more system RAM)
# 4. Higher prefetch factor (better network storage performance)
# 5. Persistent workers (reduces startup overhead on cloud)
