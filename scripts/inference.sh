#!/bin/bash

# Inference script for latent diffusion transformer
# Usage: bash scripts/inference.sh --checkpoint path/to/checkpoint.pt --data-paths path1 path2

CHECKPOINT=""
DATA_PATHS=""
FORECAST_STEPS=6
OUTPUT_DIR="outputs/predictions"
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --data-paths)
            DATA_PATHS="${@:2}"
            break
            ;;
        --forecast-steps)
            FORECAST_STEPS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CHECKPOINT" ] || [ -z "$DATA_PATHS" ]; then
    echo "Usage: bash scripts/inference.sh --checkpoint path/to/checkpoint.pt --data-paths path1 path2 ..."
    exit 1
fi

echo "Running inference..."
echo "Checkpoint: $CHECKPOINT"
echo "Forecast steps: $FORECAST_STEPS"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"

python scripts/inference.py \
    --checkpoint "$CHECKPOINT" \
    --data-paths $DATA_PATHS \
    --forecast-steps "$FORECAST_STEPS" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE"

echo "Inference completed!"
