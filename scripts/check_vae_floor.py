"""
Gate 0: is a latent forecast model viable on this data at all?

The VAE round-trip is a lossy step that every forecast passes through, so its
reconstruction error is an irreducible floor under any RMSE the latent model can
achieve. If that floor is not comfortably below the persistence baseline —
copying the last observed frame, which is the number a 10-minute nowcast has to
beat — then no amount of denoiser capacity can produce a useful forecast, and
the answer is less compression (a wider VAE, a smaller downsample, or pixel
space), not more DiT blocks.

This is the cheapest decisive fact available about the whole approach, so it is
worth one short job before committing to a full stage-1 run.

Reported per channel, in both z-scored and physical units:

    reconstruction  encode -> decode of the TARGET frame. The floor.
    persistence     the last context frame, held. The bar.
    climatology     the dataset mean, which is 0 after z-scoring. The trivial
                    forecast; RMSE ~1 by construction.

Usage:
    # untrained VAE, straight from the config (sanity check on shapes/scale)
    python scripts/check_vae_floor.py --config configs/train_config.yaml

    # the real measurement, on a stage-1 checkpoint
    python scripts/check_vae_floor.py --checkpoint /scratch/.../best_model.pt
"""

from __future__ import annotations

import argparse
import json
import logging

import torch

from src.inference import build_model_from_config, resolve_device
from src.petdata import build_dataloader
from src.training.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Checkpoint (.pt); its embedded config is used")
    src.add_argument("--config", help="Config YAML; builds an UNTRAINED VAE")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--batches", type=int, default=64)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def channel_names(config: dict) -> list[str]:
    data_cfg = config["data"]
    return list(data_cfg.get("himawari_vars", [])) + list(data_cfg.get("barra_vars", []))


def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = checkpoint.get("config")
        if not config:
            raise KeyError(f"Checkpoint {args.checkpoint} has no embedded 'config'.")
        model = build_model_from_config(config).to(device).eval()
        model.load_state_dict(checkpoint["model_state_dict"])
        label = f"checkpoint {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})"
    else:
        config = load_config(args.config)
        model  = build_model_from_config(config).to(device).eval()
        label  = f"UNTRAINED VAE from {args.config}"

    names = channel_names(config)
    C     = int(config["model"]["image_channels"])
    loader = build_dataloader(args.split, config, shuffle=False, drop_last=False)

    # Sum of squared error per channel, accumulated in float64.
    sse = {k: torch.zeros(C, dtype=torch.float64) for k in
           ("reconstruction", "persistence", "climatology")}
    n_elem = 0
    n_samples = 0

    with torch.no_grad():
        for i, (context, target) in enumerate(loader):
            if i >= args.batches:
                break
            context, target = context.to(device), target.to(device)
            B, T = target.shape[:2]

            # Deterministic round trip: mu, not a sampled z. This is what the
            # diffusion model's output is decoded from, so it is the honest
            # measure of what compression costs.
            mu, _ = model.vae.encode(target.reshape(B * T, *target.shape[2:]))
            recon = model.vae.decode(mu).view_as(target)

            persistence = context[:, -1:].expand_as(target)

            def add(key: str, pred: torch.Tensor) -> None:
                err = (pred - target).double().pow(2)
                # Sum over everything except the channel axis.
                sse[key] += err.permute(2, 0, 1, 3, 4).flatten(1).sum(dim=1).cpu()

            add("reconstruction", recon)
            add("persistence", persistence)
            add("climatology", torch.zeros_like(target))
            n_elem    += target.numel() // C
            n_samples += B

            if (i + 1) % 10 == 0:
                logger.info("  %d batches", i + 1)

    if n_elem == 0:
        raise RuntimeError(
            f"Split '{args.split}' yielded no batches — check data.splits and "
            "dataloader.drop_last."
        )

    rmse = {k: (v / n_elem).sqrt() for k, v in sse.items()}

    # Physical units, so the numbers mean something outside this codebase.
    with open(config["data"]["stats_path"]) as f:
        stats = json.load(f)

    print(f"\n{label}")
    print(f"split={args.split}  batches={min(args.batches, i + 1)}  samples={n_samples}")
    print(f"\n{'channel':<28} {'recon':>9} {'persist':>9} {'clim':>9}   {'recon (phys)':>14}")
    print("-" * 76)
    for c in range(C):
        name  = names[c] if c < len(names) else f"channel_{c}"
        scale = stats.get(name, {}).get("std", float("nan"))
        print(f"{name:<28} {rmse['reconstruction'][c]:>9.4f} "
              f"{rmse['persistence'][c]:>9.4f} {rmse['climatology'][c]:>9.4f}   "
              f"{rmse['reconstruction'][c] * scale:>14.2f}")

    c = int(config["data"].get("irradiance_channel", 0))
    floor, bar = float(rmse["reconstruction"][c]), float(rmse["persistence"][c])
    name = names[c] if c < len(names) else f"channel_{c}"

    print(f"\nVERDICT on the scored channel ({name}):")
    print(f"  VAE reconstruction floor : {floor:.4f}")
    print(f"  persistence baseline     : {bar:.4f}")
    print(f"  floor / baseline         : {floor / bar:.3f}")
    if floor < 0.5 * bar:
        print("  PASS — the floor leaves clear room under persistence.")
    elif floor < bar:
        print("  MARGINAL — the floor is below persistence but not by much. Any "
              "forecast error adds on top of it, so the model has little room.")
    else:
        print("  FAIL — reconstruction alone is already worse than copying the "
              "last frame. No denoiser can fix this; reduce the compression "
              "(wider VAE, smaller downsample) or drop the VAE entirely.")


if __name__ == "__main__":
    main()
