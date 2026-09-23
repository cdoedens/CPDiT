"""
Look at what the VAE round-trip costs, channel by channel, on one frame.

`check_vae_floor.py` reduces the same round trip to per-channel RMSE numbers.
Those say how large the error is but not *where* it lives, and a VAE can hit a
respectable RMSE while smearing exactly the structure a nowcast depends on --
cloud edges, the ragged boundary between shaded and lit ground. This plots a
single timestep so that structure is visible:

    row 1  original     the frame as the dataloader produced it
    row 2  reconstructed encode -> decode of that frame
    row 3  difference   reconstruction minus original, on a diverging scale
                        centred at zero and symmetric about it, so the sign
                        of the error reads off the colour

One column per channel. Each channel is a different physical variable, so rows
1 and 2 share a colour scale *within* a column (and only within it) -- that
shared scale is what makes the two panels comparable at a glance. The
difference row gets its own symmetric scale per column, since the errors are
typically far smaller than the field itself and would be invisible otherwise.

The round trip is deterministic: mu is decoded, not a sampled z. That matches
what the diffusion model's output is decoded from, and keeps the difference row
free of sampling noise that inference never sees.

Usage:
    # a stage-1 checkpoint -- the real measurement
    python scripts/plot_vae_reconstruction.py --checkpoint /scratch/.../best_model.pt

    # untrained VAE straight from a config, as a shapes/scale sanity check
    python scripts/plot_vae_reconstruction.py --config configs/stage1_vae.yaml

    # a different sample / a later frame in its forecast window
    python scripts/plot_vae_reconstruction.py --checkpoint ckpt.pt \
        --sample-index 4 --frame 2 --output outputs/vae_recon_s4.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.inference import build_model_from_config, resolve_device
from src.petdata import build_dataloader, channel_names
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
    p.add_argument("--sample-index", type=int, default=0,
                   help="Index of the sample within the split")
    p.add_argument("--frame", type=int, default=0,
                   help="Which frame of that sample to plot (default: 0)")
    p.add_argument("--source", default="target", choices=["target", "context"],
                   help="Take the frame from the target window (default) or the context")
    p.add_argument("--physical", action="store_true",
                   help="Un-standardise to physical units before plotting")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="outputs/vae_reconstruction.png")
    p.add_argument("--cmap", default="inferno")
    p.add_argument("--diff-cmap", default="RdBu_r")
    return p.parse_args()


def plotting_config(config: dict) -> dict:
    """
    A copy of the config pinned to one sample per batch, in split order.

    Both settings matter for `--sample-index`: the training batch size would
    make the index count batches, and with num_workers > 0 the split is sharded
    across workers and the batches arrive interleaved, so "sample 3" would
    depend on the worker count. At batch 1 with no workers the stream is
    exactly the split's own date order.
    """
    config = json.loads(json.dumps(config))   # cheap deep copy; config is plain JSON-able
    config.setdefault("dataloader", {})
    config["dataloader"]["batch_size"]  = 1
    config["dataloader"]["num_workers"] = 0
    config["dataloader"].pop("prefetch_factor", None)
    config["dataloader"].pop("persistent_workers", None)
    return config


def fetch_sample(config: dict, split: str, sample_index: int):
    """Pull one (context, target) pair from the pipeline."""
    loader = build_dataloader(split, config, shuffle=False, drop_last=False)
    for i, (context, target) in enumerate(loader):
        if i == sample_index:
            return context, target
    raise IndexError(
        f"Split '{split}' yielded fewer than {sample_index + 1} usable samples."
    )


def channel_scales(config: dict, names: list[str]) -> list[tuple[float, float]]:
    """(mean, std) per channel from the saved stats file, for physical units."""
    with open(config["data"]["stats_path"]) as f:
        stats = json.load(f)
    scales = []
    for name in names:
        entry = stats.get(name)
        if entry is None:
            raise KeyError(
                f"Channel '{name}' has no entry in {config['data']['stats_path']}; "
                f"known variables: {sorted(stats)}"
            )
        scales.append((float(entry["mean"]), float(entry["std"])))
    return scales


def plot_reconstruction(
    original: np.ndarray,   # (C, H, W)
    recon:    np.ndarray,   # (C, H, W)
    names:    list[str],
    units:    str,
    cmap:     str,
    diff_cmap: str,
    title:    str,
    output_path: str,
) -> None:
    diff = recon - original
    C    = original.shape[0]

    fig, axes = plt.subplots(3, C, figsize=(3.6 * C, 10.5), squeeze=False,
                             constrained_layout=True)

    for c in range(C):
        # Shared scale down the first two rows so the panels are comparable;
        # per-column because each channel is a different variable.
        vmin = float(min(original[c].min(), recon[c].min()))
        vmax = float(max(original[c].max(), recon[c].max()))
        # Symmetric about zero, so white is "no error" and the sign is legible.
        vdiff = float(np.abs(diff[c]).max()) or 1.0

        rows = [
            ("Original",      original[c], dict(cmap=cmap,      vmin=vmin,   vmax=vmax)),
            ("Reconstructed", recon[c],    dict(cmap=cmap,      vmin=vmin,   vmax=vmax)),
            ("Difference",    diff[c],     dict(cmap=diff_cmap, vmin=-vdiff, vmax=vdiff)),
        ]

        for row, (label, frame, kwargs) in enumerate(rows):
            ax = axes[row][c]
            ax.imshow(frame, **kwargs)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(names[c], fontsize=12)
            if c == 0:
                ax.set_ylabel(label, fontsize=12)
        # A bar per panel rather than one spanning rows 0-1: a spanning bar is
        # laid out against two axes and pushes the difference row away from
        # them, breaking the row alignment that makes the columns readable.
        # Rows 0 and 1 carry the same scale by construction, so the repeat
        # costs nothing.
        for row in range(3):
            label = units if row < 2 else f"recon - original ({units})"
            fig.colorbar(axes[row][c].images[0], ax=axes[row][c],
                         shrink=0.85, label=label)

        rmse = float(np.sqrt(np.mean(diff[c] ** 2)))
        axes[2][c].set_xlabel(f"RMSE {rmse:.4g}", fontsize=10)

    fig.suptitle(title, fontsize=14)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    logger.info("Saved %s", output_path)
    plt.close(fig)


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
        label = f"{Path(args.checkpoint).name} (epoch {checkpoint.get('epoch', '?')})"
    else:
        config = load_config(args.config)
        model  = build_model_from_config(config).to(device).eval()
        label  = f"UNTRAINED VAE from {args.config}"

    names = channel_names(config)

    logger.info("Fetching sample %d from split '%s'", args.sample_index, args.split)
    context, target = fetch_sample(plotting_config(config), args.split, args.sample_index)

    frames = target if args.source == "target" else context   # (1, T, C, H, W)
    T      = frames.shape[1]
    if not -T <= args.frame < T:
        raise IndexError(
            f"--frame {args.frame} is out of range: the {args.source} window of this "
            f"sample holds {T} frame(s)."
        )

    # (1, C, H, W) -- one timestep, kept batched so the VAE sees its usual shape.
    x = frames[:, args.frame].to(device)

    with torch.no_grad():
        mu, _ = model.vae.encode(x)
        recon = model.vae.decode(mu)

    original_np = x[0].cpu().numpy()
    recon_np    = recon[0].cpu().numpy()

    units = "z-score"
    if args.physical:
        scales      = channel_scales(config, names[: original_np.shape[0]])
        mean        = np.array([m for m, _ in scales])[:, None, None]
        std         = np.array([s for _, s in scales])[:, None, None]
        original_np = original_np * std + mean
        # The difference row is then a difference of physical values, which is
        # std * (z-score error) -- the shift cancels, as it must.
        recon_np    = recon_np * std + mean
        units       = "physical units"

    plot_reconstruction(
        original_np, recon_np,
        names=[names[c] if c < len(names) else f"channel_{c}"
               for c in range(original_np.shape[0])],
        units=units, cmap=args.cmap, diff_cmap=args.diff_cmap,
        title=f"VAE round trip -- {label}\n"
              f"{args.split} sample {args.sample_index}, {args.source} frame {args.frame}",
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
