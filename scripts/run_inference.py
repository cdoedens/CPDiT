"""
Run a trained CPDiT checkpoint over a data split and write the forecasts as NetCDF.

Forecasts are initialised at fixed local clock times, by default 11:00 and
14:00 each day (--init-times, local = UTC + --utc-offset-hours, default AEST).
Using the same times every day gives each day of the split equal weight. It
also keeps the forecasts independent: 10-minute-apart windows would share
almost all of their context. The 11:00 forecast verifies 11:10-14:00 and the
14:00 one verifies 14:10-17:00, so their lead times do not overlap.

Himawari skips one scan a day for housekeeping (02:40 UTC), which falls inside
the 14:00 AEST context window (02:10-04:00 UTC) every single day. An isolated
missing context frame -- both neighbours present -- is therefore filled by
linear interpolation in time (--max-context-gaps, default 1 per forecast) and
listed in the file's `context_interpolated` attribute. A context window with
more or longer gaps loses that forecast, and a warning is logged. Missing
truth frames never drop a forecast; they are stored as NaN.

One file per forecast, named by its initialisation time (the last observed
frame, in UTC) and grouped into year/month/day directories:

    <output-dir>/YYYY/MM/DD/YYYYMMDDTHHMM.nc

Each file holds, in physical units (the z-scoring is undone with the same
statistics the pipeline applied):

    <var>        (member, lead_time, latitude, longitude)  forecast
    <var>_obs    (lead_time, latitude, longitude)          truth (unless --no-truth)

for every channel in `channel_names(config)` -- surface_global_irradiance,
solar_elevation and RH24mean with the v0.1.0 config.

Usage (single GPU):
    python scripts/run_inference.py

Multi-GPU -- each rank takes a disjoint shard of the dates and writes its own
files, so nothing needs gathering:
    torchrun --nproc_per_node=4 scripts/run_inference.py

The config (data pipeline, splits, geometry, sampler defaults) is read from the
YAML archived next to the checkpoint (--config), not from the checkpoint.

The model forecasts `data.n_post` (= 6) frames per call, and rows of the
lead-time embedding past the trained horizon never received a gradient -- see
`rollout` in scripts/plot_forecast.py. Forecasts of --horizon-minutes (default
180, i.e. 18 frames) are therefore built autoregressively: a 6-frame block is
sampled, appended to the context window (oldest frames dropped), and the model
is called again -- 3 calls in all. Lead times run 10..180 min after the last
observed frame.

Resumable: forecasts whose file already exists are skipped (unless
--overwrite), and files are written atomically, so a job that hits walltime can
simply be resubmitted. Skipping happens after a sample is loaded, which is cheap
once the first pass has warmed the frame cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import xarray as xr
from torch.utils.data import DataLoader

# Runnable as `python scripts/run_inference.py` from anywhere, and importable
# again by spawned DataLoader workers, which re-execute this module.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference import build_model_from_config, resolve_device  # noqa: E402
from src.training.config import load_config  # noqa: E402
from src.petdata import (  # noqa: E402
    PipelineDataset,
    frame_times,
    _build_frame_pipelines,
    _parse_timestep,
    _shard_info,
    channel_names,
)

logger = logging.getLogger("run_inference")

DEFAULT_CHECKPOINT = "/scratch/er8/cd3022/CPDiT/stage2_checkpoints/best_model.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--config", default=None,
                   help="Override the config embedded in the checkpoint. By "
                        "default the checkpoint's own config is used, which is "
                        "the only one guaranteed to match its input variables.")
    p.add_argument("--horizon-minutes", type=int, default=180,
                   help="Forecast length; rolled out data.n_post frames per model call")
    p.add_argument("--output-dir", default=None,
                   help="Default: an inference/ folder beside the checkpoint, so "
                        "each model variant's forecasts stay with its weights.")
    p.add_argument("--split", default="test", help="Split name from data.splits in the checkpoint")
    p.add_argument("--start", default=None,
                   help="Override the split's start (e.g. 20250301T0000); useful for "
                        "spreading one split over several jobs")
    p.add_argument("--end", default=None, help="Override the split's end")
    p.add_argument("--init-times", default="11:00,14:00",
                   help="Comma-separated local HH:MM initialisation times, one "
                        "forecast each per day")
    p.add_argument("--utc-offset-hours", type=float, default=10.0,
                   help="Local time = UTC + this. Fixed all year (no daylight saving), "
                        "so the init times keep the same solar geometry")
    p.add_argument("--sampler", default=None, choices=["pc", "ode"],
                   help="Default: model.inference.sampler from the checkpoint")
    p.add_argument("--num-steps", type=int, default=None,
                   help="Reverse-diffusion steps. Default: model.inference.num_steps")
    p.add_argument("--members", type=int, default=10,
                   help="Ensemble members per forecast. Needs --sampler pc to vary "
                        "beyond the initial noise draw.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Forecasts per sampler call (the model sees batch_size * members)")
    p.add_argument("--vae-chunk", type=int, default=16,
                   help="Frames per VAE encoder/decoder call. Each rollout call "
                        "encodes batch_size * members * n_prior_sat full-resolution "
                        "frames, which overflows the GPU in one pass; lower this "
                        "if it still runs out of memory")
    p.add_argument("--max-context-gaps", type=int, default=1,
                   help="Isolated missing context frames to fill by interpolating "
                        "their neighbours, per forecast (0 disables)")
    p.add_argument("--workers", type=int, default=8,
                   help="DataLoader workers. Only the cold (uncached) pass benefits; "
                        "see the dataloader notes in configs/train_base.yaml")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Stop after this many forecasts (per rank); for quick checks")
    p.add_argument("--matmul-precision", default="high", choices=["highest", "high", "medium"],
                   help="torch.set_float32_matmul_precision on Ampere and newer. "
                        "'high' runs fp32 matmuls on TF32 tensor cores, as training "
                        "does; 'highest' is full fp32")
    p.add_argument("--no-truth", action="store_true", help="Do not store the observed frames")
    p.add_argument("--overwrite", action="store_true", help="Recompute existing files")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def daily_anchors(start: str, end: str, init_times: list[str], step: pd.Timedelta,
                  utc_offset: pd.Timedelta) -> list[str]:
    """
    Anchors (valid time of the first forecast frame, UTC) for every day in
    [start, end) at each local `init_times` clock time. The init time is the
    last observed frame, so the anchor is one `step` after it.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    days = pd.date_range((start_ts + utc_offset).normalize(),
                         (end_ts + utc_offset).normalize(), freq="D")
    anchors = []
    for day in days:
        for hhmm in init_times:
            anchor = day + pd.Timedelta(f"{hhmm}:00") - utc_offset + step
            if start_ts <= anchor < end_ts:
                anchors.append(anchor.strftime("%Y%m%dT%H%M"))
    return anchors


class TimestampedDataset(PipelineDataset):
    """
    `PipelineDataset` over an explicit anchor list (`anchors`, set by
    `build_loader`) instead of the split's sliding date range. It also yields
    each sample's anchor timestamp.

    The anchor is the valid time of the first forecast frame; the context is
    the `n_prior_sat` frames strictly before it (see `frame_times`). Anchors
    are sharded across DDP ranks and DataLoader workers like the base stream.

    Only the context must be usable. The `n_truth` verification frames are
    optional: a frame the archive lacks (e.g. the daily 02:40 UTC gap) or that
    is NaN (night) is stored as NaN rather than discarding the forecast, which
    would otherwise reject nearly every anchor once the window is hours long.

    Up to `max_context_gaps` isolated context frames (missing, with both
    neighbours present) are filled by linear interpolation. Without this the
    14:00 AEST forecast, whose context spans the 02:40 UTC housekeeping gap,
    would be lost every day.

    Frames are read with `_real_frame`, bypassing the dataset's own
    `data.interpolate_gap_times` fill: the gaps are filled here instead so
    every interpolated frame is reported, and a truth frame is never an
    interpolated one.
    """

    n_truth = 0                    # total verification frames wanted; set by `build_loader`
    max_context_gaps = 1           # set by `build_loader`
    anchors: list[str] = []        # set by `build_loader`

    def _sample(self, anchor: str):
        """Returns ((T_ctx + n_truth, C, H, W), interpolated context stamps),
        or None when the context is unusable."""
        stamps  = frame_times(anchor, self.context_length, max(self.n_truth, 1))
        context = [self._real_frame(stamp) for stamp in stamps[:self.context_length]]

        gaps = [k for k, frame in enumerate(context) if frame is None]
        if len(gaps) > self.max_context_gaps:
            return None
        for k in gaps:
            # Interior and isolated only: the last frame is the initial
            # condition and must be real, and a longer gap is too much to guess.
            if not 0 < k < self.context_length - 1:
                return None
            before, after = context[k - 1], context[k + 1]
            if before is None or after is None:
                return None
            context[k] = 0.5 * (before + after)

        truth = []
        for stamp in stamps[self.context_length:self.context_length + self.n_truth]:
            frame = self._real_frame(stamp)
            truth.append(torch.full_like(context[0], float("nan")) if frame is None else frame)
        return torch.stack(context + truth), [stamps[k] for k in gaps]

    def __iter__(self):
        shard_index, num_shards = _shard_info()
        anchors = self.anchors[shard_index::num_shards]
        if not anchors:
            return
        # Built here, in the consuming process: the pipeline is not fork-safe.
        self._open_pipelines()
        for anchor in anchors:
            sample = self._sample(anchor)
            if sample is None:
                logger.warning("No usable context for anchor %s; that forecast is skipped",
                               anchor)
                continue
            tensor, filled = sample
            self.n_yielded += 1
            # Joined into one string so the default collate keeps it per sample.
            yield (tensor[:self.context_length], tensor[self.context_length:], anchor,
                   ",".join(filled))


def build_loader(config: dict, split: str, batch_size: int, workers: int,
                 n_truth: int, anchors: list[str], max_context_gaps: int) -> DataLoader:
    """Mirrors `build_dataloader`, over the timestamped dataset."""
    dataset = TimestampedDataset(split, config, shuffle=False)
    dataset.n_truth = n_truth
    dataset.max_context_gaps = max_context_gaps
    dataset.anchors = anchors
    kwargs: dict = {}
    if workers > 0:
        kwargs["prefetch_factor"]         = 2
        kwargs["persistent_workers"]      = False     # single pass
        kwargs["multiprocessing_context"] = "spawn"   # pipeline is not fork-safe
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=workers,
        pin_memory=False, drop_last=False, **kwargs,
    )


def grid_coords(config: dict, stamps: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Latitude/longitude of the model grid: the Himawari frame after the same
    bounding and `crop_offset` crop `PipelineDataset._fetch_frame` applies
    (BARRA is regridded onto this grid, so it is the grid of every channel).

    Frames served from the cache carry no coordinates, so one frame is fetched
    from the archive. Pass init times: the last context frame is never
    interpolated, so it is known to be fetchable (truth frames may be missing).
    """
    data_cfg = config["data"]
    size     = config["model"].get("image_size", 256)
    o        = data_cfg.get("crop_offset", 10)
    sat, _   = _build_frame_pipelines(config)

    last_exc = None
    for stamp in stamps:
        try:
            ds = sat[stamp].isel(time=0, drop=True)
        except Exception as exc:  # noqa: BLE001 - try the next anchor
            last_exc = exc
            continue
        ds  = ds.isel(latitude=slice(o, o + size), longitude=slice(o, o + size))
        lat = ds["latitude"].values.astype(np.float64)
        lon = ds["longitude"].values.astype(np.float64)
        if (lat.size, lon.size) == (size, size):
            return lat, lon
        last_exc = ValueError(f"cropped grid is {lat.size}x{lon.size}, expected {size}x{size}")
    raise RuntimeError(f"Could not recover grid coordinates from {stamps}: {last_exc}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def output_path(output_dir: Path, init_time: pd.Timestamp) -> Path:
    return output_dir / init_time.strftime("%Y/%m/%d") / f"{init_time:%Y%m%dT%H%M}.nc"


def to_dataset(
    forecast: np.ndarray,          # (M, T, C, H, W) physical units
    truth:    np.ndarray | None,   # (T, C, H, W)    physical units
    init_time: pd.Timestamp,
    step: pd.Timedelta,
    names: list[str],
    lat: np.ndarray,
    lon: np.ndarray,
    attrs: dict,
) -> xr.Dataset:
    n_lead    = forecast.shape[1]
    lead_time = pd.to_timedelta([step * (k + 1) for k in range(n_lead)])
    coords = {
        "member":     np.arange(forecast.shape[0], dtype=np.int32),
        "lead_time":  lead_time,
        "latitude":   ("latitude", lat, {"units": "degrees_north", "standard_name": "latitude"}),
        "longitude":  ("longitude", lon, {"units": "degrees_east", "standard_name": "longitude"}),
        "init_time":  ((), np.datetime64(init_time.to_datetime64(), "ns"),
                       {"long_name": "time of the last observed context frame"}),
        "valid_time": ("lead_time", (init_time + lead_time).values),
    }

    data_vars = {}
    for c, name in enumerate(names):
        data_vars[name] = (
            ("member", "lead_time", "latitude", "longitude"),
            forecast[:, :, c].astype(np.float32),
            {"long_name": f"CPDiT forecast of {name}"},
        )
        if truth is not None:
            data_vars[f"{name}_obs"] = (
                ("lead_time", "latitude", "longitude"),
                truth[:, c].astype(np.float32),
                {"long_name": f"observed {name}"},
            )
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def write_atomic(ds: xr.Dataset, path: Path) -> None:
    """Write via a temporary name, so an interrupted job never leaves a partial
    file that the resume check would mistake for a finished one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".nc.tmp{os.getpid()}")
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(tmp, engine="netcdf4", encoding=encoding)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_distributed(device: str) -> tuple[int, int, str]:
    """Join the torchrun process group if launched under one. `_shard_info`
    then gives each rank a disjoint slice of the dates."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return 0, 1, resolve_device(device)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    return dist.get_rank(), world_size, resolve_device(device)


def load_model(checkpoint_path: str, config: dict, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    # `latent_std` is a buffer, so the load overwrote any pinned value; restore
    # it exactly as the trainer does after --resume / --init-from.
    if config["model"].get("latent_scale") is not None:
        model.latent_std.fill_(float(config["model"]["latent_scale"]))
        model.latent_scale_initialised.fill_(True)

    model.to(device).eval()
    return model, checkpoint.get("epoch")


def rollout(model, context: torch.Tensor, n_frames: int, chunk: int, members: int,
            **sample_kw) -> torch.Tensor:
    """
    Autoregressive forecast, `chunk` (= the trained `data.n_post`) frames per
    model call; `n_frames` must be a multiple of `chunk`.

    context: (B, T_ctx, C, H, W), normalised. Returns (B * members, n_frames, C, H, W)
    with the members of one forecast contiguous. Members are expanded up front
    so each follows its own trajectory instead of sharing a context.
    """
    window = context.repeat_interleave(members, dim=0)
    frames = []
    for _ in range(n_frames // chunk):
        nxt = model.sample(window, num_forecast_steps=chunk, num_samples=1, **sample_kw)
        frames.append(nxt)
        window = torch.cat([window, nxt.to(window.dtype)], dim=1)[:, -context.shape[1]:]
    return torch.cat(frames, dim=1)


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed(args.device)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [rank {rank}] %(levelname)s %(message)s",
    )
    # The sampler state stays fp32; only the matmul inputs are rounded to TF32's
    # 10 mantissa bits, with fp32 accumulation. Ignored on pre-Ampere cards.
    matmul_precision = "highest"
    if device.startswith("cuda") and torch.cuda.get_device_capability(device)[0] >= 8:
        matmul_precision = args.matmul_precision
    torch.set_float32_matmul_precision(matmul_precision)

    if args.config:
        config = load_config(args.config)
    else:
        config = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
    model, epoch = load_model(args.checkpoint, config, device)
    logger.info("Loaded %s (epoch %s) with config %s on %s", args.checkpoint, epoch,
                args.config or "embedded in the checkpoint", device)

    inf_cfg   = config["model"].get("inference", {})
    sampler   = args.sampler or inf_cfg.get("sampler", "pc")
    num_steps = args.num_steps or int(inf_cfg.get("num_steps", 100))
    members   = args.members
    n_post    = int(config["data"].get("n_post", 1))

    splits = config["data"]["splits"]
    if args.split not in splits:
        raise KeyError(f"Split {args.split!r} not in the checkpoint config: {sorted(splits)}")
    split_cfg = splits[args.split]
    if args.start:
        split_cfg["start"] = args.start
    if args.end:
        split_cfg["end"] = args.end

    value, unit = _parse_timestep(config["data"].get("sat_timestep", "10 minutes"))
    step = pd.Timedelta(value, unit=unit)
    if pd.Timedelta(minutes=args.horizon_minutes) % step:
        raise ValueError(f"--horizon-minutes must be a multiple of the frame step ({step})")
    n_lead = int(pd.Timedelta(minutes=args.horizon_minutes) // step)
    if n_lead % n_post:
        raise ValueError(f"--horizon-minutes must be a multiple of n_post * step "
                         f"({n_post} * {step}), the span of one model call")

    init_times = [t.strip() for t in args.init_times.split(",") if t.strip()]
    utc_offset = pd.Timedelta(hours=args.utc_offset_hours)
    anchors = daily_anchors(split_cfg["start"], split_cfg["end"], init_times, step, utc_offset)
    if not anchors:
        raise ValueError(f"No init times {init_times} fall inside "
                         f"{split_cfg['start']} .. {split_cfg['end']}")

    names = channel_names(config)
    with open(config["data"]["stats_path"]) as f:
        stats = json.load(f)
    # Inverse of PipelineDataset._normalise, including its epsilon.
    mean = np.array([stats[n]["mean"] for n in names], dtype=np.float32)[:, None, None]
    std  = np.array([stats[n]["std"] + 1e-8 for n in names], dtype=np.float32)[:, None, None]

    output_dir = (Path(args.output_dir) if args.output_dir
                  else Path(args.checkpoint).resolve().parent / "inference")
    attrs = {
        "title":       "CPDiT latent diffusion nowcast",
        "checkpoint":  str(args.checkpoint),
        "epoch":       -1 if epoch is None else int(epoch),
        "split":       args.split,
        "sampler":     sampler,
        "num_steps":   num_steps,
        "matmul_precision": matmul_precision,
        "config":    str(args.config or "embedded in the checkpoint"),
        "variant":     str(config.get("run", {}).get("variant", "unknown")),
        "run_id":      str(config.get("run", {}).get("run_id", "unknown")),
        "forecast_method": f"autoregressive, {n_post} frames per model call, "
                           f"{n_lead // n_post} calls",
        "init_times_local": ",".join(init_times),
        "utc_offset_hours": args.utc_offset_hours,
        "seed":        args.seed,
        "n_members":   members,
        "n_context_frames": int(config["data"]["n_prior_sat"]),
        "note": ("lead_time is measured from init_time, the last observed frame. "
                 "Fields are in the physical units of the normalisation statistics "
                 f"({config['data']['stats_path']})."),
        "created":     pd.Timestamp.now(tz="UTC").isoformat(),
    }

    logger.info(
        "Split %r %s .. %s | %d forecasts at local %s (UTC%+g) | "
        "sampler=%s steps=%d members=%d lead frames=%d (%d calls) matmul=%s | -> %s",
        args.split, split_cfg["start"], split_cfg["end"],
        len(anchors), ",".join(init_times), args.utc_offset_hours,
        sampler, num_steps, members, n_lead, n_lead // n_post, matmul_precision, output_dir,
    )

    loader = build_loader(config, args.split, args.batch_size, args.workers,
                          0 if args.no_truth else n_lead, anchors, args.max_context_gaps)
    lat = lon = None
    n_written = n_skipped = 0
    t_start = time.time()

    for batch_idx, (context, target, anchors, filled) in enumerate(loader):
        init_times = [pd.Timestamp(a) - step for a in anchors]
        paths      = [output_path(output_dir, t) for t in init_times]
        keep = [i for i, p in enumerate(paths) if args.overwrite or not p.exists()]
        n_skipped += len(paths) - len(keep)
        if args.max_samples is not None:
            keep = keep[: max(0, args.max_samples - n_written)]
        if not keep:
            if args.max_samples is not None and n_written >= args.max_samples:
                break
            continue

        if lat is None:
            lat, lon = grid_coords(config, [init_times[i].strftime("%Y%m%dT%H%M")
                                            for i in keep])

        ctx = context[keep].to(device)
        generator = torch.Generator(device=ctx.device).manual_seed(
            args.seed + 1_000_003 * rank + batch_idx
        )
        with torch.no_grad():
            pred = rollout(
                model, ctx, n_lead, n_post, members,
                sampler   = sampler,
                num_steps = num_steps,
                generator = generator,
                vae_chunk = args.vae_chunk,
            )
        # repeat_interleave order: members of one forecast are contiguous.
        pred = pred.float().cpu().numpy().reshape(len(keep), members, *pred.shape[1:])
        pred = pred * std + mean
        obs  = None if args.no_truth else target[keep].float().numpy() * std + mean

        for j, i in enumerate(keep):
            ds = to_dataset(
                pred[j], None if obs is None else obs[j],
                init_times[i], step, names, lat, lon,
                {**attrs, "context_interpolated": filled[i]},
            )
            write_atomic(ds, paths[i])
        n_written += len(keep)

        elapsed = time.time() - t_start
        logger.info(
            "batch %d: wrote %d (total %d, skipped existing %d) | %.2f forecasts/s | last %s",
            batch_idx, len(keep), n_written, n_skipped, n_written / elapsed,
            init_times[keep[-1]].strftime("%Y-%m-%d %H:%M"),
        )
        if args.max_samples is not None and n_written >= args.max_samples:
            break

    if args.workers == 0:     # with workers, the counters live in the worker processes
        loader.dataset.log_epoch_summary()
    logger.info("Done: wrote %d forecast files, skipped %d existing, in %.1f min",
                n_written, n_skipped, (time.time() - t_start) / 60)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
