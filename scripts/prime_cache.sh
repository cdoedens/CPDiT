#!/bin/bash
echo "Skipping environment set up. Make sure this is done prior to running train.sh" 


python -m src.training.train --config configs/train_config.yaml --prime-cache 

echo "Priming completed!"