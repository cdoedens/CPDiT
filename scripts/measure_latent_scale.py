"""
Measure the VAE latent standard deviation, for pinning `model.latent_scale`.

Where this sits in the workflow
-------------------------------
    stage 1  train the VAE alone          configs/stage1_vae.yaml
    THIS     measure the latent std  -->  paste into model.latent_scale
    stage 2  train the diffusion model    configs/train_config.yaml

Why it has to be pinned rather than tracked
-------------------------------------------
The diffusion process assumes it is corrupting roughly unit-variance data, and
the sampler starts its reverse chain from N(0, I). Both are only true if the
divisor applied to the latents actually matches their spread. While the VAE is
training that number moves, so the model tracks it by EMA — but an EMA lags, and
in stage 2 the VAE is frozen and there is nothing to track. Measuring it once
and freezing it removes the moving target entirely, which is what Stable
Diffusion's fixed 0.18215 factor does.

Usage:
    python scripts/measure_latent_scale.py \
        --checkpoint /scratch/.../stage1_checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import logging

import torch

from src.inference import build_model_from_config, resolve_device
from src.petdata import build_dataloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Stage-1 checkpoint (.pt); its embedded config is used")
    p.add_argument("--split", default="train", choices=["train", "val", "test"],
                   help="Split to measure over. Use the training split: this "
                        "number describes the data the diffusion model fits.")
    p.add_argument("--batches", type=int, default=64,
                   help="Batches to accumulate over")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if not config:
        raise KeyError(f"Checkpoint {args.checkpoint} has no embedded 'config'.")

    model = build_model_from_config(config).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])

    loader = build_dataloader(args.split, config, shuffle=False, drop_last=False)

    # Pooled variance over every latent element seen, accumulated in float64.
    # The mean is tracked as well rather than assumed to be zero: the statistic
    # that matters for scaling is the spread about the actual mean, and nothing
    # in the VAE forces the latent to be centred.
    n = 0
    total = 0.0
    total_sq = 0.0

    with torch.no_grad():
        for i, (context, target) in enumerate(loader):
            if i >= args.batches:
                break
            frames = torch.cat([context, target], dim=1).to(device)
            B, T = frames.shape[:2]
            mu, _ = model.vae.encode(frames.reshape(B * T, *frames.shape[2:]))
            mu = mu.double()
            n        += mu.numel()
            total    += float(mu.sum())
            total_sq += float(mu.pow(2).sum())
            if (i + 1) % 10 == 0:
                logger.info("  %d batches, running std %.6f", i + 1,
                            (total_sq / n - (total / n) ** 2) ** 0.5)

    if n == 0:
        raise RuntimeError(
            f"Split '{args.split}' yielded no batches — check data.splits and "
            "dataloader.drop_last."
        )

    mean = total / n
    std  = (total_sq / n - mean ** 2) ** 0.5

    logger.info("Latent elements: %d over %d batches", n, min(args.batches, i + 1))
    logger.info("Latent mean: %.6f", mean)
    logger.info("Latent std:  %.6f", std)
    logger.info("EMA value carried in the checkpoint: %.6f",
                float(model.latent_std))
    print("\nSet this in configs/train_config.yaml for stage 2:\n")
    print(f"model:\n  latent_scale: {std:.6g}\n")

    if abs(mean) > 0.5 * std:
        logger.warning(
            "The latent mean (%.4f) is large relative to its std (%.4f). Scaling "
            "only divides, so the diffusion process will see off-centre data and "
            "the N(0, I) prior will be biased. Consider a stronger KL "
            "(training.vae_beta) in stage 1.", mean, std,
        )


if __name__ == "__main__":
    main()
