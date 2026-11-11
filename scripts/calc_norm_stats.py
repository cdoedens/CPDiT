"""
Normalisation statistics (mean and standard deviation) for every input variable.

    python scripts/calc_norm_stats.py                      # all of configs/inputs/*.yaml
    python scripts/calc_norm_stats.py --inputs configs/inputs/atmos.yaml
    python scripts/calc_norm_stats.py --only-missing       # keep existing entries
    qsub scripts/calc_norm_stats.pbs                       # as a job: it needs more
                                                           # memory than a login node allows

Covers the union of the variables in the given inputs files: Himawari solar
variables, raw Himawari bands and BARRA variables, including the derived BARRA
ones (KI, TCD).

Frames are drawn at random from the training split, at the same hours training
uses (`data.skip_utc_hours` is respected), and every variable is read by the
code path that builds training frames, on the model grid -- so the statistics
describe what the network actually sees, in raw physical units. A frame whose
irradiance has NaN (night) is skipped, as training would skip it. BARRA is
taken at the hour the frame falls in.

The result is MERGED into the stats file: the computed entries are added or
replaced and every other entry is left alone. Statistics are part of the
frame-cache key for the channels a config uses, so REPLACING a variable's entry
starts a new cache for every variant using it; `--only-missing` computes only
the variables the file does not have yet and so keeps existing caches valid.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.training.config import INPUT_KEYS, load_config      # noqa: E402

logger = logging.getLogger("calc_norm_stats")

_dataset = None      # per-process PipelineDataset, built by _init_worker


def union_inputs(paths: list[Path]) -> dict[str, list[str]]:
    """Every variable named by the inputs files, per input key, each once."""
    union: dict[str, list[str]] = {key: [] for key in INPUT_KEYS}
    for path in paths:
        with open(path) as f:
            data = (yaml.safe_load(f) or {}).get("data") or {}
        for key in INPUT_KEYS:
            for var in data.get(key) or []:
                if var not in union[key]:
                    union[key].append(var)
    return union


def candidate_stamps(config: dict, n: int, seed: int) -> list[str]:
    from src.petdata import blocked_utc_hours

    split = config["data"]["splits"]["train"]
    stamps = pd.date_range(pd.Timestamp(split["start"]), pd.Timestamp(split["end"]),
                           freq="10min", inclusive="left")
    blocked = blocked_utc_hours(config["data"])
    stamps = stamps[~stamps.hour.isin(sorted(blocked))]
    rng = np.random.default_rng(seed)
    # Oversample: some timestamps are missing from the archive or partly dark.
    picks = rng.choice(len(stamps), size=min(len(stamps), 4 * n), replace=False)
    return [stamps[i].strftime("%Y%m%dT%H%M") for i in picks]


def _init_worker(config: dict) -> None:
    """Build this process's archive pipelines (they are not fork-safe)."""
    global _dataset
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from src.petdata import PipelineDataset

    # Used only for its fetch path: no cache, and no statistics to normalise by.
    _dataset = PipelineDataset("train", config, shuffle=False, stats={})
    _dataset._open_pipelines()


def frame_fields(stamp: str):
    """
    {variable: (H, W) array in raw physical units} for one frame on the model
    grid, or None when the frame is unusable. Needs `_init_worker` to have run
    in this process.
    """
    from src.petdata import bracketing_hours, select_region

    ds, cfg = _dataset, _dataset.config
    data_cfg = cfg["data"]
    try:
        sat = ds._sat_pipe[stamp].isel(time=0, drop=True)
        sat = select_region(sat, cfg)
        # NaN at night; it also supplies the model grid for everything else.
        gate = (data_cfg["himawari_vars"] or [None])[0]
        if gate is not None and np.isnan(sat[gate].values).any():
            return None
        fields = {v: sat[v].values for v in data_cfg["himawari_vars"]}

        if ds.bands:
            bands = ds._fetch_bands(stamp, sat["latitude"].values, sat["longitude"].values)
            fields.update({b: bands[b].values for b in ds.bands})

        if ds._bar_source.variables:
            hour, _, _ = bracketing_hours(stamp)
            bar = ds._bar_source[hour].interp(
                latitude=sat.latitude, longitude=sat.longitude, method="nearest"
            )
            fields.update({v: bar[v].values for v in ds._bar_source.variables})
    except Exception as exc:  # noqa: BLE001 - skip missing timestamps
        logger.info("skip %s: %s: %s", stamp, type(exc).__name__, str(exc)[:200])
        return None
    return fields


def _frame_sums(stamp: str):
    """
    {variable: (sum, sum of squares, count)} over one frame's finite pixels,
    or None when the frame is unusable.
    """
    fields = frame_fields(stamp)
    if fields is None:
        return None

    sums = {}
    for name, values in fields.items():
        v = np.asarray(values, dtype=np.float64)
        v = v[np.isfinite(v)]
        sums[name] = (float(v.sum()), float(np.square(v).sum()), int(v.size))
    return sums


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=str(REPO / "configs" / "train_base.yaml"))
    parser.add_argument("--inputs", nargs="+", default=None,
                        help="Inputs files whose variables to compute "
                             "(default: every configs/inputs/*.yaml).")
    parser.add_argument("--n-frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8,
                        help="Processes fetching frames. More than ~10 is slower "
                             "(see scripts/prime_cache.sh).")
    parser.add_argument("--only-missing", action="store_true",
                        help="Compute only variables not yet in the stats file.")
    parser.add_argument("--output", default=None,
                        help="Stats file to merge into (default: data.stats_path).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    inputs = ([Path(p) for p in args.inputs] if args.inputs
              else sorted((REPO / "configs" / "inputs").glob("*.yaml")))
    if not inputs:
        raise SystemExit("No inputs files found.")

    # The missing statistics are what this script is here to produce.
    config = load_config(args.config, inputs=inputs[0], require_stats=False)
    stats_path = Path(args.output or config["data"]["stats_path"])
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    union = union_inputs(inputs)
    if args.only_missing:
        union = {key: [v for v in names if v not in stats] for key, names in union.items()}
    variables = [v for key in INPUT_KEYS for v in union[key]]
    if not variables:
        raise SystemExit(f"Nothing to compute: {stats_path} covers every variable.")
    logger.info("Computing statistics for %s", variables)

    # One dataset reading the whole union. The irradiance frame stays in, first,
    # as the night-time gate and the source of the model grid.
    gate = config["data"]["himawari_vars"][:1]
    union["himawari_vars"] = gate + [v for v in union["himawari_vars"] if v not in gate]
    config["data"].update(union)
    config["dataloader"]["cache_dir"] = None
    config["model"].pop("image_channels", None)   # the dataset's check does not apply

    total = {v: np.zeros(3) for v in variables}
    n_ok = 0
    stamps = candidate_stamps(config, args.n_frames, args.seed)
    workers = max(1, args.workers)
    # Spawned, not forked: see build_dataloader in src/petdata.
    pool = ProcessPoolExecutor(workers, mp_context=get_context("spawn"),
                               initializer=_init_worker, initargs=(config,))
    try:
        for sums in pool.map(_frame_sums, stamps):
            if sums is None:
                continue
            for v in variables:
                total[v] += sums[v]
            n_ok += 1
            if n_ok % 25 == 0:
                logger.info("%d / %d frames", n_ok, args.n_frames)
            if n_ok >= args.n_frames:
                break
    except BrokenProcessPool as exc:
        # A killed worker leaves no traceback. MEASURED: a worker grows to
        # ~3.5 GB within ten frames, and the Gadi login nodes kill anything
        # over 4 GB per user -- so this is what running there looks like.
        raise SystemExit(
            "A worker process was killed, most likely for memory. Run this as "
            "a job (qsub scripts/calc_norm_stats.pbs), not on a login node."
        ) from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    if n_ok == 0:
        raise SystemExit("No usable frames found.")
    if n_ok < args.n_frames:
        logger.warning("Only %d of the %d requested frames were usable.", n_ok, args.n_frames)

    for v in variables:
        s, sq, count = total[v]
        if count == 0:
            raise SystemExit(f"{v} had no finite values in {n_ok} frames.")
        mean = s / count
        std = float(np.sqrt(max(sq / count - mean ** 2, 0.0)))
        stats[v] = {"mean": float(mean), "std": std}
        logger.info("%s: mean=%.6g std=%.6g", v, mean, std)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=4))
    logger.info("Merged statistics for %d variables from %d frames into %s",
                len(variables), n_ok, stats_path)


if __name__ == "__main__":
    main()
