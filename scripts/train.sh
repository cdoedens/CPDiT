#!/bin/bash

# Training script for latent diffusion transformer
# Usage: bash scripts/train.sh [config_path]

CONFIG_PATH=${1:-configs/train_config.yaml}
DEVICE=${2:-cuda}

echo "Starting training with config: $CONFIG_PATH"
echo "Device: $DEVICE"

source hpc_setup.sh

# Run training
python -m src.training.train \
    --config "$CONFIG_PATH" \
    --device "$DEVICE"

# torchrun --nproc_per_node=4 -m src_testing.training.train \
#     --config "$CONFIG_PATH" \

echo "Training completed!"
