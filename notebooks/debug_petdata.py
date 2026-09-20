import torch
import numpy as np
import sys
from src_testing.petdata import build_dataloader
from src_testing.training.config import load_config

config = load_config("/home/548/cd3022/repos/CPDiT/configs/experimental_config.yaml")
loader = build_dataloader("train", config, shuffle=False)

iterator = iter(loader)
i=0
for context, forecast in iterator:
    i+=1
    # tensor = torch.tensor(sample, dtype=torch.float32)

    # Skip samples containing NaN or Inf (e.g. night-time masked pixels)
    if np.isnan(context).any() or np.isinf(context).any():
        print("NANs found, skipping")
        continue
    print("No NANs in data")
    if i == 10:
        break