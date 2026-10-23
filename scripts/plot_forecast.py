"""
Plot a 3x6 forecast panel: prior context, truth, and model prediction.

Row 1 (top)    the six irradiance frames t-60min .. t-10min (the last context frames)
Row 2 (middle) the irradiance truth at t+0, +10, +20, +50, +110, +170 min
Row 3 (bottom) the model's forecast at those same lead times

Usage:
    python scripts/plot_forecast.py --checkpoint /path/to/ckpt.pt \
        --split val --sample-index 0 --output outputs/forecast.png

    # or a specific forecast time (UTC): context ends t0-10min, forecast starts at t0
    python scripts/plot_forecast.py --checkpoint /path/to/ckpt.pt \
        --datetime "2024-03-05 02:30" --output outputs/forecast.png

The config (data pipeline, image geometry, forecast horizon, ...) is read straight
out of the checkpoint, matching how `load_model_from_checkpoint` reconstructs the
model elsewhere in this repo -- nothing here can drift out of sync with what the
checkpoint was actually trained with.

The model forecasts `data.n_post` frames per call (t+0 .. t+(n_post-1)*step).
Lead times past that horizon are reached by rolling the model forward
autoregressively -- see `rollout`. Three 6-frame calls cover t+0 .. t+170, and the
last plotted frame is the last frame of the third call.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.inference import Forecaster, build_model_from_config, resolve_device
from src.petdata import (
    PipelineDataset, _parse_timestep, blocked_utc_hours, build_dataloader, channel_names,
    frame_times,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Lead times plotted in each row, in minutes relative to the forecast anchor.
# The context tensor holds strictly-earlier frames (last one is t-10) and
# `frame_times` puts the first forecast frame AT the anchor (t+0), so the forecast
# row starts at t+0 and three 6-frame calls end exactly on t+170.
CONTEXT_OFFSETS_MIN  = [-60, -50, -40, -30, -20, -10]
FORECAST_OFFSETS_MIN = [0, 10, 20, 50, 110, 170]

N_PANELS = len(FORECAST_OFFSETS_MIN)   # frames per row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Path to a training checkpoint (.pt)")
    p.add_argument("--split", default="val", choices=["train", "val", "test"],
                   help="Which data split to draw the sample from. With --datetime it only "
                        "selects which frame cache is used, so pass the split the date falls in")
    p.add_argument("--datetime", nargs="+", default=None, metavar="DATETIME",
                   help="Forecast anchor time t0 (UTC), e.g. 2024-03-05 02:30 or 20240305T0230 "
                        "(quotes optional). "
                        "The context ends at t0-10min and the first forecast frame is t0. "
                        "Must fall on the frame grid. Mutually exclusive with --sample-index.")
    p.add_argument("--sample-index", type=int, default=None,
                   help="Index of the sample within the split to plot (default 0 when "
                        "--datetime is not given)")
    p.add_argument("--irradiance-channel", type=int, default=None,
                   help="Channel to plot (default: data.irradiance_channel from the config, else 0)")
    p.add_argument("--sampler", default=None, choices=["pc", "ode"],
                   help="Override the checkpoint's inference sampler (deterministic 'ode' by default)")
    p.add_argument("--num-steps", type=int, default=None,
                   help="Reverse-diffusion integration steps (default: from the checkpoint config)")
    p.add_argument("--forecast-chunk", type=int, default=None,
                   help="Frames per model call when rolling out (default: data.n_post from the "
                        "checkpoint, i.e. the horizon it was actually trained on)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="outputs/forecast.png")
    p.add_argument("--cmap", default="inferno")
    args = p.parse_args()
    if args.datetime is not None:
        args.datetime = " ".join(args.datetime)
    if args.datetime is not None and args.sample_index is not None:
        p.error("--datetime and --sample-index are mutually exclusive")
    return args


def load_model_and_forecaster(checkpoint_path: str, device: str, num_steps, sampler):
    """Mirrors `src.inference.load_model_from_checkpoint`, but lets the CLI
    override the sampler/step-count without touching that shared helper."""
    device     = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config")
    if not config:
        raise KeyError(f"Checkpoint {checkpoint_path} has no embedded 'config'.")

    model   = build_model_from_config(config)
    inf_cfg = config["model"].get("inference", {})

    forecaster = Forecaster(
        model, checkpoint_path=str(checkpoint_path), device=device,
        num_steps=num_steps or inf_cfg.get("num_steps", 100),
        sampler=sampler or inf_cfg.get("sampler", "ode"),
    )
    return config, forecaster


def panel_indices(offsets_min: list[int], step_minutes: int) -> list[int]:
    """
    Turn lead times in minutes into frame indices.

    Index `k` is the frame at t + k*step: non-negative values index the target
    (and prediction) tensors, negative ones index the context tensor from its
    end, since the context runs up to -- but not including -- the anchor.
    """
    indices = []
    for offset in offsets_min:
        if offset % step_minutes:
            raise ValueError(
                f"Lead time t{offset:+d}min is not a multiple of the {step_minutes}min "
                f"frame spacing, so no frame lands on it."
            )
        indices.append(offset // step_minutes)
    return indices


def plotting_config(config: dict, step_minutes: int) -> dict:
    """
    A copy of the training config adjusted for plotting a single sample.

    `data.n_post` is forced to cover the longest lead time in
    `FORECAST_OFFSETS_MIN`, regardless of what the checkpoint trained with --
    the dataloader reads frames straight off the archive/cache, so this is a
    query-time choice, not a retraining one. It only widens the *truth* window;
    the model is still asked for its trained horizon per call. The truth row is
    a sparse selection out of that window, not every frame in it.

    The loader is also pinned to batch_size 1 and num_workers 0. Both matter
    for `--sample-index`: the training batch size would make the index count
    batches, and with workers > 0 the split is sharded across them and the
    batches arrive interleaved, so "sample 3" would depend on the worker
    count. At batch 1 with no workers the stream is exactly the split's own
    date order.
    """
    config = json.loads(json.dumps(config))   # cheap deep copy; config is plain JSON-able
    config["data"]["n_post"] = max(panel_indices(FORECAST_OFFSETS_MIN, step_minutes)) + 1
    config.setdefault("dataloader", {})
    config["dataloader"]["batch_size"]  = 1
    config["dataloader"]["num_workers"] = 0
    config["dataloader"].pop("prefetch_factor", None)
    config["dataloader"].pop("persistent_workers", None)
    return config


def rollout(forecaster, context: torch.Tensor, n_frames: int, chunk: int) -> torch.Tensor:
    """
    Forecast `n_frames` frames, `chunk` at a time, feeding predictions back in.

    A model trained with `data.n_post = k` has only ever produced k frames per
    call: the denoiser conditions each forecast frame on `lead_embedder[i]`, and
    rows i >= k of that embedding never received a gradient, so asking for a
    longer window in one call returns frames conditioned on random init. Only
    lead times the model was trained on are requested here; everything beyond
    them is reached by sliding the context window forward over the model's own
    output, which is the trained map applied repeatedly.

    Errors compound across the rollout -- that is inherent to the trained
    horizon, not to this implementation.

    (`Forecaster.forecast_sequence(autoregressive=True)` chunks by the
    *architectural* `max_forecast_steps` rather than the trained horizon, so it
    would collapse this into a single direct call.)
    """
    context_length = forecaster.model.context_length
    window    = context
    frames    = []
    remaining = n_frames

    while remaining > 0:
        step     = min(chunk, remaining)
        forecast = forecaster.forecast_deterministic(window, num_steps=step)
        frames.append(forecast)
        # Slide the window forward, keeping the context length the denoiser
        # was built for (its input convolution is sized for a fixed T_ctx).
        window     = torch.cat([window, forecast.to(window.device)], dim=1)[:, -context_length:]
        remaining -= step

    return torch.cat(frames, dim=1)


def fetch_sample(config: dict, split: str, sample_index: int):
    """Pull one (context, target) pair from the pipeline, and its anchor time."""
    loader = build_dataloader(split, config, shuffle=False, drop_last=False)
    for i, (context, target) in enumerate(loader):
        if i == sample_index:
            return context, target, loader.dataset.last_anchor
    raise IndexError(
        f"Split '{split}' yielded fewer than {sample_index + 1} usable samples."
    )


def fetch_sample_at(config: dict, split: str, when: str, step_minutes: int):
    """Pull the (context, target) pair anchored at a specific time."""
    anchor = pd.Timestamp(when)
    if anchor.tzinfo is not None:
        anchor = anchor.tz_convert("UTC").tz_localize(None)
    if anchor.minute % step_minutes or anchor.second or anchor.microsecond:
        raise ValueError(
            f"--datetime {when!r} is not on the {step_minutes}min frame grid "
            f"(minute must be a multiple of {step_minutes})."
        )
    blocked = blocked_utc_hours(config["data"])
    if anchor.hour in blocked:
        raise ValueError(
            f"{anchor:%Y-%m-%d %H:%M} UTC falls in data.skip_utc_hours (local night), "
            f"where the irradiance field is absent."
        )

    stamp   = anchor.strftime("%Y%m%dT%H%M")
    dataset = PipelineDataset(split, config)
    pair    = dataset.sample_at(stamp, allow_missing_target=True)
    if pair is None:
        window = frame_times(stamp, dataset.context_length, 0, step_minutes)
        bad    = [t for t in window if dataset.cache.exists(t)[1]]
        detail = (f" Known-bad frames: {', '.join(bad)}." if bad else "")
        raise ValueError(
            f"Cannot forecast from {anchor:%Y-%m-%d %H:%M} UTC: a context frame "
            f"({window[0]} .. {window[-1]}) is missing or NaN (e.g. before the archive "
            f"starts, or a gap).{detail}"
        )
    context, target = pair
    return context[None], target[None], stamp     # add the batch dimension


def irradiance_stats(config: dict, channel: int) -> tuple[str, float, float]:
    """Look up (name, mean, std) for the plotted channel from the saved stats file."""
    var_names = channel_names(config)
    if channel >= len(var_names):
        raise IndexError(
            f"Channel {channel} has no matching variable name; known channels: {var_names}"
        )
    var_name = var_names[channel]

    with open(config["data"]["stats_path"]) as f:
        stats = json.load(f)
    entry = stats[var_name]
    return var_name, float(entry["mean"]), float(entry["std"])


def plot_forecast(
    context: np.ndarray,   # (N_PANELS, H, W)  physical units
    truth:   np.ndarray,   # (N_PANELS, H, W)
    pred:    np.ndarray,   # (N_PANELS, H, W)
    var_name: str,
    anchor_label: str,
    cmap: str,
    output_path: str,
) -> None:
    # Scaled on the observations only. Folding the prediction in lets a single
    # bad forecast frame set the range and flatten every real frame to one
    # colour, which hides exactly what the figure is for. The forecast is then
    # drawn on the same scale as the truth it is compared against, and anything
    # outside it saturates (visibly) rather than rescaling the panel.
    # nan-aware: a truth frame the archive lacks is all-NaN and drawn blank.
    observed = np.concatenate([context, truth])
    vmin, vmax = float(np.nanmin(observed)), float(np.nanmax(observed))

    fig, axes = plt.subplots(3, N_PANELS, figsize=(2.4 * N_PANELS, 7.5), constrained_layout=True)

    rows = [
        ("Context",  context, CONTEXT_OFFSETS_MIN),
        ("Truth",    truth,   FORECAST_OFFSETS_MIN),
        ("Forecast", pred,    FORECAST_OFFSETS_MIN),
    ]

    im = None
    for row, (label, frames, offsets) in enumerate(rows):
        for col in range(N_PANELS):
            ax = axes[row, col]
            im = ax.imshow(frames[col], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            sign = "+" if offsets[col] >= 0 else ""
            no_data = np.isnan(frames[col]).all()
            ax.set_title(f"t{sign}{offsets[col]}min" + ("\n(no data)" if no_data else ""),
                         fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=12)

    fig.colorbar(im, ax=axes, shrink=0.8, label=var_name)
    fig.suptitle(
        f"Irradiance forecast: context / truth / prediction\nt0 = {anchor_label}", fontsize=14
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    logger.info("Saved %s", output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    config, forecaster = load_model_and_forecaster(
        args.checkpoint, args.device, args.num_steps, args.sampler
    )

    channel = args.irradiance_channel
    if channel is None:
        channel = config["data"].get("irradiance_channel", 0)

    step_value, step_unit = _parse_timestep(config["data"].get("sat_timestep", "10 minutes"))
    step_minutes = step_value if step_unit.startswith("min") else step_value * 60

    context_idx  = panel_indices(CONTEXT_OFFSETS_MIN, step_minutes)
    forecast_idx = panel_indices(FORECAST_OFFSETS_MIN, step_minutes)
    n_forecast   = max(forecast_idx) + 1        # frames needed to reach the last lead time
    n_prior      = -min(context_idx)            # frames needed before the anchor (0 if none)

    plot_cfg = plotting_config(config, step_minutes)

    if args.datetime is not None:
        logger.info("Fetching sample anchored at %s", args.datetime)
        context, target, anchor = fetch_sample_at(plot_cfg, args.split, args.datetime, step_minutes)
    else:
        index = args.sample_index or 0
        logger.info("Fetching sample %d from split '%s'", index, args.split)
        context, target, anchor = fetch_sample(plot_cfg, args.split, index)
    anchor_label = f"{pd.Timestamp(anchor):%Y-%m-%d %H:%M} UTC"
    logger.info("Forecast anchor t0 = %s", anchor_label)

    context_length = context.shape[1]
    if context_length < n_prior:
        raise ValueError(
            f"The checkpoint's context is {context_length} frames, fewer than the "
            f"{n_prior} the context row plots (back to t-{n_prior * step_minutes}min). "
            f"Re-run with a model trained on at least {n_prior} prior frames."
        )

    chunk = args.forecast_chunk or int(config["data"].get("n_post", 1))
    n_calls = -(-n_forecast // chunk)     # ceil
    logger.info(
        "Forecasting %d frames as %d call(s) of up to %d frame(s), %d integration steps each",
        n_forecast, n_calls, chunk, forecaster.num_steps,
    )
    if n_calls > 1:
        logger.info(
            "Rolling the model forward autoregressively: it was trained on a %d-frame "
            "horizon (%dmin), and the plot reaches t+%dmin. Errors compound with lead time.",
            chunk, chunk * step_minutes, max(FORECAST_OFFSETS_MIN),
        )
    pred = rollout(forecaster, context, n_forecast, chunk)

    var_name, mean, std = irradiance_stats(config, channel)

    prior_np    = context[0, :, channel].numpy() * std + mean   # t-context_length*step .. t-step
    target_np   = target[0, :, channel].numpy() * std + mean    # t+0 .. t+(n_forecast-1)*step
    forecast_np = pred[0, :, channel].numpy() * std + mean      # same window, predicted

    # Negative indices reach back into the context frames, non-negative ones sit
    # in the forecast window -- which is where the anchor frame t+0 lives.
    context_np = np.stack([prior_np[i] if i < 0 else target_np[i] for i in context_idx])
    truth_np   = target_np[forecast_idx]
    pred_np    = forecast_np[forecast_idx]

    plot_forecast(
        context_np, truth_np, pred_np,
        var_name=var_name, anchor_label=anchor_label, cmap=args.cmap, output_path=args.output,
    )


if __name__ == "__main__":
    main()
