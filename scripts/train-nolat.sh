#!/bin/bash

# Training script for latent diffusion transformer
# Usage: bash scripts/train.sh [config_path]

CONFIG_PATH=${1:-configs/experimental_config.yaml}
DEVICE=${2:-cuda}

echo "Starting training with config: $CONFIG_PATH"
echo "Device: $DEVICE"

source hpc_setup.sh

# Run training
# python -m src_nolat.training.train \
#     --config "$CONFIG_PATH" \

torchrun --nproc_per_node=1 -m src_nolat.training.train \
    --config "$CONFIG_PATH" \

echo "Training completed!"