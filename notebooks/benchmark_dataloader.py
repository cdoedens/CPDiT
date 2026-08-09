"""
Dataloader and GPU utilisation benchmark.

Measures:
  - Dataloader throughput (samples/sec, batches/sec)
  - GPU utilisation and memory during a simulated training loop
  - Time breakdown: data loading vs forward pass vs backward pass

Run interactively on a GPU node before submitting a full training job.
Usage:
    python scripts/benchmark.py --config configs/your_config.yaml --batches 50
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data import build_dataloader
from src.models.vae import VariationalAutoencoder
from src.models.latent_diffusion import LatentDiffusionTransformer

# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

class CudaTimer:
    """Measures wall time with CUDA synchronisation so GPU work is included."""

    def __init__(self, device):
        self.device   = device
        self.start_t  = None

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.start_t = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self.start_t


def gpu_stats(device) -> str:
    if device.type != "cuda":
        return "N/A (no CUDA)"
    alloc   = torch.cuda.memory_allocated(device)  / 1024**3
    reserved = torch.cuda.memory_reserved(device)  / 1024**3
    return f"alloc={alloc:.2f}GB  reserved={reserved:.2f}GB"


# ---------------------------------------------------------------------------
# Benchmark 1: raw dataloader throughput (no GPU)
# ---------------------------------------------------------------------------

def benchmark_dataloader(config: dict, num_batches: int) -> None:
    print("\n" + "="*60)
    print("  BENCHMARK 1: Raw DataLoader throughput (CPU only)")
    print("="*60)

    loader = build_dataloader("train", config, shuffle=False)
    batch_size = loader.batch_size

    # Warm up one batch (opens Zarr stores in workers)
    print("  Warming up workers...")
    it = iter(loader)
    next(it)

    times   = []
    samples = 0

    print(f"  Timing {num_batches} batches...")
    for i, (context, forecast) in enumerate(it):
        if i >= num_batches:
            break
        t0 = time.perf_counter()
        # Force materialisation (tensors are already on CPU)
        _ = context.numpy()
        _ = forecast.numpy()
        times.append(time.perf_counter() - t0)
        samples += batch_size

    # Report
    total_time  = sum(times)
    mean_batch  = np.mean(times)
    p50         = np.percentile(times, 50)
    p95         = np.percentile(times, 95)
    throughput  = samples / total_time

    print(f"\n  Batches timed    : {len(times)}")
    print(f"  Total time       : {total_time:.2f}s")
    print(f"  Mean batch time  : {mean_batch*1000:.1f}ms")
    print(f"  P50 batch time   : {p50*1000:.1f}ms")
    print(f"  P95 batch time   : {p95*1000:.1f}ms")
    print(f"  Throughput       : {throughput:.1f} samples/sec")
    print(f"  Context shape    : {context.shape}")
    print(f"  Forecast shape   : {forecast.shape}")


# ---------------------------------------------------------------------------
# Benchmark 2: VAE stage 1 (reconstruction loop)
# ---------------------------------------------------------------------------

def benchmark_vae(config: dict, device: torch.device, num_batches: int) -> None:
    print("\n" + "="*60)
    print("  BENCHMARK 2: VAE Stage 1 training loop")
    print("="*60)

    loader   = build_dataloader("train", config, shuffle=False)
    model_cfg = config["model"]

    vae = VariationalAutoencoder(
        image_channels  = model_cfg["image_channels"],
        latent_channels = model_cfg["latent_channels"],
        hidden_dim      = model_cfg["vae_hidden_dim"],
        image_size      = model_cfg["image_size"],
    ).to(device)

    optimiser = torch.optim.Adam(vae.parameters(), lr=1e-4)

    data_times    = []
    forward_times = []
    backward_times = []

    print(f"  Device: {device}")
    print(f"  GPU memory before model load: {gpu_stats(device)}")

    it = iter(loader)

    # Warm-up pass (not timed)
    print("  Running warm-up pass...")
    context, forecast = next(it)
    B, T, C, H, W = context.shape
    x = context[:, 0].to(device)  # single frame: (B, C, H, W)
    x_recon, mu, logvar = vae(x)
    loss, _, _ = vae.vae_loss(x, x_recon, mu, logvar)
    loss.backward()
    optimiser.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print(f"  GPU memory after warm-up:     {gpu_stats(device)}")
    print(f"  Input shape per step: {x.shape}")
    print(f"  Timing {num_batches} batches...\n")

    for i in range(num_batches):
        # --- Data loading ---
        with CudaTimer(device) as data_t:
            context, forecast = next(it)
            B, T, C, H, W = context.shape
            # Flatten time into batch: train on all frames independently
            x = context.view(B * T, C, H, W).to(device, non_blocking=True)

        # --- Forward ---
        with CudaTimer(device) as fwd_t:
            x_recon, mu, logvar = vae(x)
            loss, recon_loss, kl_loss = vae.vae_loss(x, x_recon, mu, logvar)

        # --- Backward ---
        with CudaTimer(device) as bwd_t:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

        data_times.append(data_t.elapsed)
        forward_times.append(fwd_t.elapsed)
        backward_times.append(bwd_t.elapsed)

        if (i + 1) % 10 == 0:
            print(
                f"  [{i+1:3d}/{num_batches}] "
                f"loss={loss.item():.4f}  "
                f"recon={recon_loss.item():.4f}  "
                f"kl={kl_loss.item():.4f}  "
                f"| data={data_times[-1]*1000:.1f}ms  "
                f"fwd={forward_times[-1]*1000:.1f}ms  "
                f"bwd={backward_times[-1]*1000:.1f}ms  "
                f"| {gpu_stats(device)}"
            )

    _report_timing("VAE Stage 1", data_times, forward_times, backward_times,
                   loader.batch_size, device)


# ---------------------------------------------------------------------------
# Benchmark 3: Diffusion stage 2 (frozen VAE)
# ---------------------------------------------------------------------------

def benchmark_diffusion(config: dict, device: torch.device, num_batches: int) -> None:
    print("\n" + "="*60)
    print("  BENCHMARK 3: Diffusion Stage 2 training loop")
    print("="*60)

    loader    = build_dataloader("train", config, shuffle=False)
    model_cfg = config["model"]

    model = LatentDiffusionTransformer(
        image_channels         = model_cfg["image_channels"],
        image_size             = model_cfg["image_size"],
        latent_channels        = model_cfg["latent_channels"],
        vae_hidden_dim         = model_cfg["vae_hidden_dim"],
        num_transformer_layers = model_cfg["num_transformer_layers"],
        num_heads              = model_cfg["num_heads"],
        feedforward_dim        = model_cfg["feedforward_dim"],
        transformer_dim        = model_cfg["transformer_dim"],
        num_diffusion_steps    = model_cfg["num_diffusion_steps"],
        denoiser_hidden_dim    = model_cfg["denoiser_hidden_dim"],
        dropout                = model_cfg["dropout"],
    ).to(device)

    model.freeze_vae()
    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    data_times     = []
    forward_times  = []
    backward_times = []

    print(f"  Device: {device}")
    print(f"  GPU memory after model load: {gpu_stats(device)}")

    it = iter(loader)

    # Warm-up
    print("  Running warm-up pass...")
    context, forecast = next(it)
    context  = context.to(device)
    forecast = forecast.to(device)
    loss, _ = model(context, forecast)
    loss.backward()
    optimiser.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print(f"  GPU memory after warm-up:    {gpu_stats(device)}")
    print(f"  Context shape:  {context.shape}")
    print(f"  Forecast shape: {forecast.shape}")
    print(f"  Timing {num_batches} batches...\n")

    for i in range(num_batches):
        # --- Data loading ---
        with CudaTimer(device) as data_t:
            context, forecast = next(it)
            context  = context.to(device, non_blocking=True)
            forecast = forecast.to(device, non_blocking=True)

        # --- Forward ---
        with CudaTimer(device) as fwd_t:
            loss, _ = model(context, forecast)

        # --- Backward ---
        with CudaTimer(device) as bwd_t:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

        data_times.append(data_t.elapsed)
        forward_times.append(fwd_t.elapsed)
        backward_times.append(bwd_t.elapsed)

        if (i + 1) % 10 == 0:
            print(
                f"  [{i+1:3d}/{num_batches}] "
                f"loss={loss.item():.4f}  "
                f"| data={data_times[-1]*1000:.1f}ms  "
                f"fwd={forward_times[-1]*1000:.1f}ms  "
                f"bwd={backward_times[-1]*1000:.1f}ms  "
                f"| {gpu_stats(device)}"
            )

    _report_timing("Diffusion Stage 2", data_times, forward_times, backward_times,
                   loader.batch_size, device)


# ---------------------------------------------------------------------------
# Shared reporting
# ---------------------------------------------------------------------------

def _report_timing(
    label:          str,
    data_times:     list,
    forward_times:  list,
    backward_times: list,
    batch_size:     int,
    device:         torch.device,
) -> None:
    total_step = [d + f + b for d, f, b in
                  zip(data_times, forward_times, backward_times)]
    n          = len(total_step)
    throughput = (n * batch_size) / sum(total_step)

    data_pct = 100 * sum(data_times)    / sum(total_step)
    fwd_pct  = 100 * sum(forward_times) / sum(total_step)
    bwd_pct  = 100 * sum(backward_times)/ sum(total_step)

    print(f"\n  --- {label} summary ({n} batches) ---")
    print(f"  Mean step time   : {np.mean(total_step)*1000:.1f}ms")
    print(f"  P95 step time    : {np.percentile(total_step, 95)*1000:.1f}ms")
    print(f"  Throughput       : {throughput:.1f} samples/sec")
    print(f"  Time breakdown   : data={data_pct:.1f}%  "
          f"forward={fwd_pct:.1f}%  backward={bwd_pct:.1f}%")

    if data_pct > 20:
        print(
            f"\n  ⚠  Data loading is {data_pct:.0f}% of step time. "
            f"Consider increasing num_workers or prefetch_factor."
        )
    else:
        print(f"\n  ✓  Data loading is only {data_pct:.0f}% of step time — GPU is the bottleneck (good).")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"  Peak GPU memory  : {peak:.2f} GB")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True, help="Path to YAML config file")
    parser.add_argument("--batches", type=int, default=50,
                        help="Number of batches to time per benchmark")
    parser.add_argument("--stage",   type=int, default=None, choices=[1, 2],
                        help="Only run benchmark for this stage (default: run both)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
        zarr_dir = Path(config["data"]["data_dir"])
        config["data"]["zarr_paths"] = {
            "train": sorted(zarr_dir.glob(config["data"]["train"])),
            "val":   sorted(zarr_dir.glob(config["data"]["val"])),
            "test":  sorted(zarr_dir.glob(config["data"]["test"])),
        }
        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")

    benchmark_dataloader(config, args.batches)

    if args.stage is None or args.stage == 1:
        benchmark_vae(config, device, args.batches)

    if args.stage is None or args.stage == 2:
        benchmark_diffusion(config, device, args.batches)


if __name__ == "__main__":
    main()