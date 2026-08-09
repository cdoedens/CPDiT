# scripts/profile_dataloader.py
import time
import torch
from src.data import build_dataloader
from src.training.config import load_config

config = load_config("configs/train_config.yaml")
loader = build_dataloader("train", config, shuffle=False)

print("Profiling 20 batches...\n")

load_times    = []
compute_times = []

iterator = iter(loader)

for i in range(20):
    t0 = time.perf_counter()
    context, forecast = next(iterator)
    context  = context.cuda(non_blocking=True)
    forecast = forecast.cuda(non_blocking=True)
    t1 = time.perf_counter()

    # Simulate a fast forward pass so GPU time is measurable
    _ = context.mean()
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    load_times.append(t1 - t0)
    compute_times.append(t2 - t1)
    print(f"Batch {i:02d}:  load={t1-t0:.3f}s  gpu={t2-t1:.3f}s")

print(f"\nMean load time : {sum(load_times)/len(load_times):.3f}s")
print(f"Mean GPU time  : {sum(compute_times)/len(compute_times):.3f}s")
print(f"Bottleneck     : {'DATA LOADING' if sum(load_times) > sum(compute_times) else 'GPU COMPUTE'}")
