#!/bin/bash

# Aggressive optimization training script for DEKR on RTX 4060
# This version pushes the limits for maximum speed

cd /home/wner/DEKR

# Checkpoint path
CKPT=output/coco_kpt/hrnet_dekr/w32_rink_512/checkpoint.pth.tar

# Aggressively optimized training command
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 \
python tools/train.py \
  --cfg experiments/rink/w32/w32_rink_512.yaml \
  TRAIN.RESUME True \
  TRAIN.CHECKPOINT $CKPT \
  TRAIN.IMAGES_PER_GPU 16 \
  WORKERS 12 \
  AMP True \
  TF32 True \
  CHANNELS_LAST True \
  PIN_MEMORY True \
  PREFETCH_FACTOR 12 \
  PERSISTENT_WORKERS True \
  FILE_SYSTEM_SHARING True \
  PRINT_FREQ 10

# Aggressive optimizations:
# 1. Batch size increased to 16 (100% increase from original 8)
# 2. Workers increased to 12 (maximum for 16-thread CPU)
# 3. PREFETCH_FACTOR increased to 12 (maximum data prefetching)
# 4. OMP_NUM_THREADS set to 12 (leaves 4 threads for system)
# 5. Reduced PRINT_FREQ to 10 (less I/O overhead)
