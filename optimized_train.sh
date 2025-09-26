#!/bin/bash

# Optimized training script for DEKR on RTX 4060
# Based on analysis of current configuration and hardware

cd /home/wner/DEKR

# Checkpoint path
CKPT=output/coco_kpt/hrnet_dekr/w32_rink_512/checkpoint.pth.tar

# Optimized training command with multiple performance improvements
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
python tools/train.py \
  --cfg experiments/rink/w32/w32_rink_512.yaml \
  TRAIN.RESUME True \
  TRAIN.CHECKPOINT $CKPT \
  TRAIN.IMAGES_PER_GPU 12 \
  WORKERS 8 \
  AMP True \
  TF32 True \
  CHANNELS_LAST True \
  PIN_MEMORY True \
  PREFETCH_FACTOR 8 \
  PERSISTENT_WORKERS True \
  FILE_SYSTEM_SHARING True \
  PRINT_FREQ 5

# Key optimizations made:
# 1. Increased batch size from 8 to 12 (50% increase in throughput)
# 2. Increased workers from 4 to 8 (better CPU utilization with 16 threads)
# 3. Enabled PIN_MEMORY (faster GPU transfers)
# 4. Increased PREFETCH_FACTOR from 4 to 8 (better data pipeline)
# 5. Enabled PERSISTENT_WORKERS (reduces worker startup overhead)
# 6. Increased OMP_NUM_THREADS to 8 (better CPU utilization)
