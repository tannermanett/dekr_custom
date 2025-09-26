#!/bin/bash

# Conservative training script for DEKR - fixes shared memory issues
# Use this if the optimized script still has problems

cd /home/wner/DEKR

# Checkpoint path
CKPT=output/coco_kpt/hrnet_dekr/w32_rink_512/checkpoint.pth.tar

# Conservative training command - minimal memory usage
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
python tools/train.py \
  --cfg experiments/rink/w32/w32_rink_512.yaml \
  TRAIN.RESUME True \
  TRAIN.CHECKPOINT $CKPT \
  TRAIN.IMAGES_PER_GPU 4 \
  WORKERS 0 \
  AMP True \
  TF32 True \
  CHANNELS_LAST False \
  PIN_MEMORY False \
  PREFETCH_FACTOR 1 \
  PERSISTENT_WORKERS False \
  FILE_SYSTEM_SHARING False \
  PRINT_FREQ 10

# Ultra-conservative settings:
# 1. Batch size 4 (very safe)
# 2. No workers (single-threaded data loading)
# 3. All memory optimizations disabled
# 4. Minimal threading
