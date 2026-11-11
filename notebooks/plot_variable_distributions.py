"""
Histogram of every input variable's distribution, in raw physical units, to
judge whether a plain z-score is an appropriate normalisation for it.

    python notebooks/plot_variable_distributions.py                # all configs/inputs/*.yaml
    python notebooks/plot_variable_distributions.py --inputs configs/inputs/atmos.yaml
    python notebooks/plot_variable_distributions.py --replot       # redraw from saved samples

Run the fetch inside a job (interactive or batch): it needs more memory than a
login node allows (4 GB per user; one worker reaches ~3.5 GB within ten
frames). `--replot` needs no job.

Frames are sampled exactly as scripts/calc_norm_stats.py samples them (random
training-split frames at training hours, read by the dataloader's own fetch
path on the model grid), and a random subset of each frame's pixels is kept.
The samples are saved next to the figure so it can be redrawn without fetching.

Each panel shows the variable's histogram with a normal curve of the same mean
and standard deviation laid over it: where the two agree, z-scoring gives the
network a roughly unit-normal input. The panel title carries the numbers that
say when it does not:

    skew   0 for a symmetric distribution; |skew| > ~1 is strongly lopsided
    kurt   excess kurtosis; 0 for a normal, large = heavy tails
    range  the smallest and largest sample in z-score units -- the extremes
           the network would actually be fed
    zero   fraction of samples exactly 0 (a spike a z-score cannot spread out)
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import calc_norm_stats as cns                                # noqa: E402
from src.training.config import INPUT_KEYS, load_config      # noqa: E402

logger = logging.getLogger("plot_variable_distributions")

_PIXELS_PER_FRAME = 20_000

# One series per panel, so one hue; reference marks stay in neutral ink.
_SURFACE, _INK, _INK_MUTED, _HUE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"


def _frame_sample(args):
    """A random subset of one frame's pixels per variable, or None."""
    stamp, seed = args
    fields = cns.frame_fields(stamp)
    if fields is None:
        return None
    rng = np.random.default_rng(seed)
    out = {}
    for name, values in fields.items():
        v = np.asarray(values, dtype=np.float32).ravel()
        v = v[np.isfinite(v)]
        if v.size > _PIXELS_PER_FRAME:
            v = rng.choice(v, size=_PIXELS_PER_FRAME, replace=False)
        out[name] = v
    return out


def collect(args) -> dict[str, np.ndarray]:
    """Fetch the samples: {variable: 1-D array of raw values}."""
    inputs = ([Path(p) for p in args.inputs] if args.inputs
              else sorted((REPO / "configs" / "inputs").glob("*.yaml")))
    if not inputs:
        raise SystemExit("No inputs files found.")

    config = load_config(args.config, inputs=inputs[0], require_stats=False)
    union = cns.union_inputs(inputs)
    variables = [v for key in INPUT_KEYS for v in union[key]]
    # The irradiance frame stays in, first: the night-time gate and the grid.
    gate = config["data"]["himawari_vars"][:1]
    union["himawari_vars"] = gate + [v for v in union["himawari_vars"] if v not in gate]
    config["data"].update(union)
    config["dataloader"]["cache_dir"] = None
    config["model"].pop("image_channels", None)
    logger.info("Sampling %s", variables)

    stamps = cns.candidate_stamps(config, args.n_frames, args.seed)
    chunks: dict[str, list[np.ndarray]] = {v: [] for v in variables}
    n_ok = 0
    jobs = [(stamp, args.seed + k) for k, stamp in enumerate(stamps)]
    pool = None
    if args.workers > 0:
        pool = ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"),
                                   initializer=cns._init_worker, initargs=(config,))
        results = pool.map(_frame_sample, jobs)
    else:
        cns._init_worker(config)          # in this process: least memory
        results = map(_frame_sample, jobs)
    try:
        for sample in results:
            if sample is None:
                continue
            for v in variables:
                chunks[v].append(sample[v])
            n_ok += 1
            if n_ok % 10 == 0:
                logger.info("%d / %d frames", n_ok, args.n_frames)
            if n_ok >= args.n_frames:
                break
    except BrokenProcessPool as exc:
        raise SystemExit(
            "A worker process was killed, most likely for memory. Run this in "
            "a job, not on a login node."
        ) from exc
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    if n_ok == 0:
        raise SystemExit("No usable frames found.")
    logger.info("Sampled %d frames", n_ok)
    return {v: np.concatenate(chunks[v]) for v in variables}


def describe(v: np.ndarray) -> dict[str, float]:
    v = v.astype(np.float64)
    mean, std = v.mean(), v.std()
    z = (v - mean) / (std + 1e-12)
    return {
        "mean": mean, "std": std,
        "skew": float(np.mean(z ** 3)),
        "kurt": float(np.mean(z ** 4) - 3.0),
        "zmin": float(z.min()), "zmax": float(z.max()),
        "zero": float(np.mean(v == 0)),
    }


def plot(samples: dict[str, np.ndarray], path: Path, bins: int, log_y: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(samples)
    ncols = min(4, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.5 * nrows),
                             squeeze=False, facecolor=_SURFACE)

    for ax, name in zip(axes.ravel(), names):
        v = samples[name]
        d = describe(v)
        ax.set_facecolor(_SURFACE)
        density, _, _ = ax.hist(v, bins=bins, density=True, color=_HUE, linewidth=0,
                                label="samples")
        lo, hi = v.min(), v.max()
        x = np.linspace(lo, hi, 400)
        pdf = np.exp(-0.5 * ((x - d["mean"]) / d["std"]) ** 2) / (d["std"] * np.sqrt(2 * np.pi))
        ax.plot(x, pdf, color=_INK, linewidth=1.2,
                label="normal, same mean and std")
        ax.axvline(d["mean"], color=_INK_MUTED, linewidth=0.8, linestyle="--",
                   label="mean")
        if log_y:
            ax.set_yscale("log")
            # Down to the emptiest occupied bin, not to the normal's far tail.
            ax.set_ylim(bottom=density[density > 0].min() / 2)

        title = (f"{name}\n"
                 f"skew {d['skew']:+.2f}   kurt {d['kurt']:+.2f}   "
                 f"range [{d['zmin']:.1f}, {d['zmax']:.1f}] σ")
        if d["zero"] > 0.001:
            title += f"   zero {100 * d['zero']:.0f}%"
        ax.set_title(title, fontsize=9, color=_INK, loc="left")
        ax.set_xlabel(f"mean {d['mean']:.4g}, std {d['std']:.4g}  (raw units)",
                      fontsize=8, color=_INK_MUTED)
        ax.tick_params(labelsize=8, colors=_INK_MUTED, length=3)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c8c2")
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)

    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=3, frameon=False,
               fontsize=9, labelcolor=_INK_MUTED)
    fig.suptitle("Input variable distributions" + ("  (log density)" if log_y else ""),
                 x=0.01, ha="left", fontsize=13, color=_INK)
    fig.supylabel("density", fontsize=9, color=_INK_MUTED)
    fig.tight_layout(rect=(0.01, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    logger.info("Wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=str(REPO / "configs" / "train_base.yaml"))
    parser.add_argument("--inputs", nargs="+", default=None,
                        help="Inputs files whose variables to plot "
                             "(default: every configs/inputs/*.yaml).")
    parser.add_argument("--n-frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8,
                        help="Processes fetching frames; 0 fetches in this process.")
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--output", default=str(REPO / "outputs" / "variable_distributions.png"),
                        help="Figure path. A log-density twin (*_log.png) and "
                             "the samples (*.npz) are written beside it.")
    parser.add_argument("--replot", action="store_true",
                        help="Redraw from the saved samples instead of fetching.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    out = Path(args.output)
    npz = out.with_suffix(".npz")
    if args.replot:
        if not npz.exists():
            raise SystemExit(f"No saved samples at {npz}; run without --replot first.")
        with np.load(npz) as f:
            samples = {k: f[k] for k in f.files}
    else:
        samples = collect(args)
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, **samples)
        logger.info("Saved samples to %s", npz)

    for name, v in samples.items():
        d = describe(v)
        logger.info("%-28s n=%-9d mean=%-11.5g std=%-11.5g skew=%+.2f kurt=%+.2f "
                    "z-range=[%.1f, %.1f] zero=%.1f%%", name, v.size, d["mean"], d["std"],
                    d["skew"], d["kurt"], d["zmin"], d["zmax"], 100 * d["zero"])

    # Linear density shows the bulk; log density shows the tails.
    plot(samples, out, args.bins, log_y=False)
    plot(samples, out.with_name(out.stem + "_log" + out.suffix), args.bins, log_y=True)


if __name__ == "__main__":
    main()
