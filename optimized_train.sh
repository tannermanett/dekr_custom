#!/bin/bash

# Optimized training script for DEKR on RTX 4060
# Based on analysis of current configuration and hardware

cd /home/wner/DEKR

# Checkpoint path
CKPT=output/coco_kpt/hrnet_dekr/w32_rink_512/checkpoint.pth.tar

# Optimized training command with shared memory fixes
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python tools/train.py \
  --cfg experiments/rink/w32/w32_rink_512.yaml \
  TRAIN.RESUME True \
  TRAIN.CHECKPOINT $CKPT \
  TRAIN.IMAGES_PER_GPU 8 \
  WORKERS 2 \
  AMP True \
  TF32 True \
  CHANNELS_LAST True \
  PIN_MEMORY False \
  PREFETCH_FACTOR 2 \
  PERSISTENT_WORKERS False \
  FILE_SYSTEM_SHARING False \
  PRINT_FREQ 5

# Key fixes for shared memory issues:
# 1. Reduced batch size from 12 to 8 (prevents OOM)
# 2. Reduced workers from 8 to 2 (prevents shared memory exhaustion)
# 3. Disabled PIN_MEMORY (reduces memory pressure)
# 4. Reduced PREFETCH_FACTOR from 8 to 2 (less memory per worker)
# 5. Disabled PERSISTENT_WORKERS (reduces memory overhead)
# 6. Reduced OMP_NUM_THREADS to 4 (prevents CPU oversubscription)
# 7. Disabled FILE_SYSTEM_SHARING (reduces shared memory usage)
