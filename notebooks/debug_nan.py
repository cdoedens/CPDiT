# scripts/debug_dataloader.py
import torch
import numpy as np
from src.data import build_dataloader
from src.training.config import load_config

config = load_config("configs/train_config.yaml")

# Force num_workers=0 so everything runs in the main process —
# this eliminates worker crashes as a variable.
config["data"]["num_workers"] = 0

loader = build_dataloader("train", config, shuffle=False)

print("Checking first 10 batches with num_workers=0...\n")
for i, (context, forecast) in enumerate(loader):
    images      = torch.cat([context, forecast], dim=1)
    flat_images = images.reshape(-1, *images.shape[2:])

    has_nan = torch.isnan(flat_images).any().item()
    has_inf = torch.isinf(flat_images).any().item()

    print(f"Batch {i:02d}:  nan={has_nan}  inf={has_inf}  "
          f"range=[{flat_images.min():.3f}, {flat_images.max():.3f}]  "
          f"shape={tuple(flat_images.shape)}")

    if has_nan or has_inf:
        # Find which channel is bad
        for c in range(flat_images.shape[1]):
            ch = flat_images[:, c]
            print(f"  channel {c}:  nan={torch.isnan(ch).any().item()}  "
                  f"inf={torch.isinf(ch).any().item()}  "
                  f"range=[{ch.min():.3f}, {ch.max():.3f}]")
        break

    if i >= 9:
        break

print("\nDone.")
