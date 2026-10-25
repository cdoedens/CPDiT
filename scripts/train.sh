#!/bin/bash

# Training script for latent diffusion transformer


echo "Skipping environment set up. Make sure this is done prior to running train.sh" 

NPROC="${NPROC:-1}"

echo $NPROC

if [ "$NPROC" -gt 1 ]; then
    torchrun --nproc_per_node="$NPROC" -m src.training.train \
        --config configs/train_base.yaml "$@"   # pass --inputs configs/inputs/<variant>.yaml
else
    # Single GPU / CPU: no torchrun needed.
    python -m src.training.train --config configs/train_base.yaml "$@"   # pass --inputs configs/inputs/<variant>.yaml
fi

echo "Training completed!"