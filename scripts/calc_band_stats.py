"""
Normalisation statistics for the raw Himawari bands in a config.

    python scripts/calc_band_stats.py --config configs/train_base.yaml \
        --inputs configs/inputs/<variant>.yaml [--n-frames 300]

Frames are drawn at random from the training split, at the same hours training
uses (`data.skip_utc_hours` is respected), and sampled onto the model grid by
exactly the code path that builds training frames -- so the statistics describe
what the network actually sees, in raw physical units.

The result is MERGED into the stats file: band entries are added or replaced
and every other entry is left alone. Statistics are part of the frame-cache
key only for the channels a config uses, so adding band entries does not
invalidate caches built by configs without them.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.petdata import PipelineDataset, blocked_utc_hours   # noqa: E402
from src.training.config import load_config                  # noqa: E402

logger = logging.getLogger("calc_band_stats")


def candidate_stamps(config: dict, n: int, seed: int) -> list[str]:
    split = config["data"]["splits"]["train"]
    stamps = pd.date_range(pd.Timestamp(split["start"]), pd.Timestamp(split["end"]),
                           freq="10min", inclusive="left")
    blocked = blocked_utc_hours(config["data"])
    stamps = stamps[~stamps.hour.isin(sorted(blocked))]
    rng = np.random.default_rng(seed)
    # Oversample: some timestamps are missing from the archive.
    picks = rng.choice(len(stamps), size=min(len(stamps), 3 * n), replace=False)
    return [stamps[i].strftime("%Y%m%dT%H%M") for i in picks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--inputs", default=None,
                        help="Inputs file whose himawari_bands to compute.")
    parser.add_argument("--n-frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None,
                        help="Stats file to merge into (default: data.stats_path).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # The missing statistics are what this script is here to produce.
    config = load_config(args.config, inputs=args.inputs, require_stats=False)
    config["dataloader"]["cache_dir"] = None
    bands = list(config["data"].get("himawari_bands", []))
    if not bands:
        raise SystemExit("data.himawari_bands is empty; nothing to compute.")

    # The dataset is only used for its fetch path. Its own channel-count check
    # does not apply here.
    config["model"].pop("image_channels", None)
    ds = PipelineDataset("train", config, shuffle=False)
    ds._open_pipelines()
    size, o = ds.image_size, ds.crop_offset

    total = {b: 0.0 for b in bands}
    total_sq = {b: 0.0 for b in bands}
    count = {b: 0 for b in bands}
    n_ok = 0
    for stamp in candidate_stamps(config, args.n_frames, args.seed):
        if n_ok >= args.n_frames:
            break
        try:
            # The irradiance frame supplies the model grid and, being NaN at
            # night, keeps the sample to frames training would accept.
            sat = ds._sat_pipe[stamp].isel(time=0, drop=True)
            sat = sat.isel(latitude=slice(o, o + size), longitude=slice(o, o + size))
            if np.isnan(sat[ds.config["data"]["himawari_vars"][0]].values).any():
                continue
            raw = ds._fetch_bands(stamp, sat["latitude"].values, sat["longitude"].values)
        except Exception as exc:  # noqa: BLE001 - skip missing timestamps
            logger.info("skip %s: %s: %s", stamp, type(exc).__name__, exc)
            continue
        for b in bands:
            v = raw[b].values.astype(np.float64)
            v = v[np.isfinite(v)]
            total[b] += v.sum()
            total_sq[b] += np.square(v).sum()
            count[b] += v.size
        n_ok += 1
        if n_ok % 25 == 0:
            logger.info("%d / %d frames", n_ok, args.n_frames)

    if n_ok == 0:
        raise SystemExit("No usable frames found.")

    stats_path = Path(args.output or config["data"]["stats_path"])
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for b in bands:
        mean = total[b] / count[b]
        std = float(np.sqrt(max(total_sq[b] / count[b] - mean ** 2, 0.0)))
        stats[b] = {"mean": float(mean), "std": std}
        logger.info("%s: mean=%.5g std=%.5g", b, mean, std)

    stats_path.write_text(json.dumps(stats, indent=4))
    logger.info("Merged statistics for %d bands from %d frames into %s",
                len(bands), n_ok, stats_path)


if __name__ == "__main__":
    main()
