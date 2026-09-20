"""
scripts/benchmark.py

Quick diagnostics for DataLoader throughput and GPU utilisation.
Uses Trainer directly so benchmark behaviour is always identical to
real training, with no duplicated forward-pass logic.

Usage:
    python notebooks/benchmark.py --config configs/train_config.yaml
    python notebooks/benchmark.py --config configs/train_config.yaml --batches 100 --stage 1
    python notebooks/benchmark.py --config configs/train_config.yaml --dataloader-only
"""

import argparse
import time

import numpy as np
import torch
import yaml

from src.training.train import Trainer


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class CudaTimer:
    def __init__(self, device: torch.device):
        self.device  = device
        self.elapsed = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self._t0


def gpu_mem(device: torch.device) -> str:
    if device.type != "cuda":
        return "N/A"
    alloc    = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device)  / 1024 ** 3
    return f"alloc={alloc:.2f} GB  reserved={reserved:.2f} GB"


def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Benchmark 1 — raw DataLoader throughput (no GPU work)
# ---------------------------------------------------------------------------

def benchmark_dataloader(trainer: Trainer, num_batches: int) -> None:
    print_header(f"BENCHMARK 1: Raw DataLoader throughput  (split=train)")

    train_loader, _ = trainer.setup_data()
    batch_size      = train_loader.batch_size
    it              = iter(train_loader)

    print("  Warming up workers (first batch discarded) ...")
    context, forecast = next(it)

    images = torch.cat([context, forecast], dim=1)
    flat   = images.reshape(-1, *images.shape[2:])
    print(f"  Context shape    : {context.shape}   dtype={context.dtype}")
    print(f"  Forecast shape   : {forecast.shape}   dtype={forecast.dtype}")
    print(f"  Flat VAE input   : {flat.shape}  (B*T, C, H, W)")
    print(f"  Effective batch  : {flat.shape[0]} frames/step")
    print(f"\n  Timing {num_batches} batches ...")

    times = []
    for i, (ctx, fct) in enumerate(it):
        if i >= num_batches:
            break
        t0 = time.perf_counter()
        _ = ctx.numpy()
        _ = fct.numpy()
        times.append(time.perf_counter() - t0)

    if not times:
        print("  ⚠  No batches returned — check dataset size and batch_size.")
        return

    total      = sum(times)
    throughput = (len(times) * batch_size) / total

    print(f"\n  Batches timed    : {len(times)}")
    print(f"  Total time       : {total:.2f} s")
    print(f"  Mean batch       : {np.mean(times)*1000:.1f} ms")
    print(f"  P50  batch       : {np.percentile(times, 50)*1000:.1f} ms")
    print(f"  P95  batch       : {np.percentile(times, 95)*1000:.1f} ms")
    print(f"  Throughput       : {throughput:.1f} samples/sec")

    if np.percentile(times, 95) > 3 * np.median(times):
        print(
            "\n  ⚠  Large P50→P95 gap — occasional slow reads.\n"
            "     Check Zarr chunk alignment against your batch/window size."
        )


# ---------------------------------------------------------------------------
# Benchmark 2 — GPU training loop (forward + backward)
# ---------------------------------------------------------------------------

def benchmark_training(trainer: Trainer, num_batches: int) -> None:
    device = torch.device(trainer.device)
    print_header(
        f"BENCHMARK 2: GPU training loop  "
        f"(split=train, stage={trainer.stage})"
    )

    n_params = sum(p.numel() for p in trainer.model.parameters()) / 1e6
    print(f"  Model parameters : {n_params:.1f} M")
    print(f"  Stage            : {trainer.stage}")
    print(f"  Mixed precision  : {trainer.use_amp}")
    print(f"  Device           : {device}")
    print(f"  GPU memory (init): {gpu_mem(device)}")

    train_loader, _ = trainer.setup_data()
    trainer.model.train()
    it = iter(train_loader)
    
    # Warm-up pass — not timed
    print("\n  Running warm-up pass ...")
    ctx, fct = next(it)
    ctx = ctx.to(device, non_blocking=True)
    fct = fct.to(device, non_blocking=True)
    
    trainer.optimizer.zero_grad()
    
    if trainer.stage == 1:
        # _stage1_step handles backward internally
        trainer._stage1_step(ctx, fct)
    else:
        loss = trainer._stage2_step(ctx, fct)
        trainer.scaler.scale(loss).backward()
    
    trainer.optimizer.zero_grad(set_to_none=True)


    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    print(f"  GPU memory (post-warmup): {gpu_mem(device)}")
    print(f"\n  Timing {num_batches} batches ...")

    data_times, fwd_times, bwd_times = [], [], []

    for i in range(num_batches):
        # ---- forward + backward ----
        with CudaTimer(device) as t_fwd:
            if trainer.stage == 1:
                loss = trainer._stage1_step(ctx, fct)
            else:
                loss = trainer._stage2_step(ctx, fct)
        
        with CudaTimer(device) as t_bwd:
            trainer.optimizer.zero_grad(set_to_none=True)
            if trainer.stage == 2:
                # stage 1 already backpropped inside _stage1_step
                trainer.scaler.scale(loss).backward()
            if trainer.gradient_clip_norm > 0:
                trainer.scaler.unscale_(trainer.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainer.model.parameters(), trainer.gradient_clip_norm
                )
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()

        data_times.append(t_data.elapsed)
        fwd_times.append(t_fwd.elapsed)
        bwd_times.append(t_bwd.elapsed)

        if (i + 1) % 10 == 0:
            print(
                f"  [{i+1:3d}/{num_batches}]  "
                f"loss={loss.item():.4f}  "
                f"data={t_data.elapsed*1000:.1f}ms  "
                f"fwd={t_fwd.elapsed*1000:.1f}ms  "
                f"bwd={t_bwd.elapsed*1000:.1f}ms  "
                f"| {gpu_mem(device)}"
            )

    # ---- Summary ----
    step_times = [d + f + b for d, f, b in zip(data_times, fwd_times, bwd_times)]
    total      = sum(step_times)
    n          = len(step_times)
    throughput = (n * train_loader.batch_size) / total
    data_pct   = 100 * sum(data_times) / total
    fwd_pct    = 100 * sum(fwd_times)  / total
    bwd_pct    = 100 * sum(bwd_times)  / total

    print(f"\n  --- Summary ({n} batches, batch_size={train_loader.batch_size}) ---")
    print(f"  Throughput       : {throughput:.1f} samples/sec")
    print(f"  Mean step        : {np.mean(step_times)*1000:.1f} ms")
    print(f"  P50  step        : {np.percentile(step_times, 50)*1000:.1f} ms")
    print(f"  P95  step        : {np.percentile(step_times, 95)*1000:.1f} ms")
    print(f"  Time split       : data={data_pct:.1f}%  "
          f"fwd={fwd_pct:.1f}%  bwd={bwd_pct:.1f}%")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        print(f"  Peak GPU memory  : {peak:.2f} GB")

    if data_pct > 20:
        print(
            f"\n  ⚠  DataLoader is {data_pct:.0f}% of step time — GPU is starved.\n"
            f"     Try increasing num_workers or prefetch_factor in the config."
        )
    else:
        print(
            f"\n  ✓  DataLoader is only {data_pct:.0f}% of step time — "
            f"GPU is the bottleneck (good)."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          required=True)
    parser.add_argument("--batches",         type=int, default=50)
    parser.add_argument("--stage",           type=int, default=None,
                        help="Override training.stage in config (1 or 2).")
    parser.add_argument("--dataloader-only", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.stage is not None:
        config["training"]["stage"] = args.stage

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU    : {props.name}")
        print(f"VRAM   : {props.total_memory / 1024**3:.1f} GB")

    trainer = Trainer(config, device=str(device))

    benchmark_dataloader(trainer, args.batches)

    if not args.dataloader_only:
        benchmark_training(trainer, args.batches)


if __name__ == "__main__":
    main()
