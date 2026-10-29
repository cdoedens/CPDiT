"""
Score forecast directories written by run_inference.py against their truth.

    python scripts/compare_ensembles.py BASE_DIR OTHER_DIR [OTHER_DIR ...]

Only forecasts present in every directory are scored, so a short re-run of a
few dates can be compared like-for-like with a full-year baseline. Per lead
time, over all scored pixels with valid truth:

    bias      ensemble mean minus truth
    rmse      RMSE of the ensemble mean
    spread    sqrt of the mean unbiased ensemble variance
    ratio     spread * sqrt((M+1)/M) / rmse; ~1 is calibrated, <1 overconfident
              (Fortin et al. 2014)
    crps      fair ensemble CRPS (Ferro 2014)
    sharp     mean |spatial gradient| of the members / that of the truth; <1
              means the forecast fields are smoother than the observations

Needs no GPU or pyearthtools.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

METRICS = ("bias", "rmse", "spread", "ratio", "crps", "sharp")


def _grad(a: np.ndarray) -> float:
    """Mean absolute spatial gradient over the last two axes, NaN-aware."""
    gy = np.abs(np.diff(a, axis=-2))[..., :, :-1]
    gx = np.abs(np.diff(a, axis=-1))[..., :-1, :]
    return float(np.nanmean(gx + gy))


def score_file(path: Path, var: str) -> dict[str, np.ndarray]:
    """Per-lead sums for one forecast, so files can be pooled by pixel count."""
    with xr.open_dataset(path) as ds:
        fc_all  = ds[var].values                          # (M, L, H, W)
        obs_all = ds[f"{var}_obs"].values                 # (L, H, W)
    M, L = fc_all.shape[:2]
    keys = ("n", "err", "se", "var", "crps", "grad_fc", "grad_obs")
    out = {k: np.zeros(L) for k in keys}
    out["M"] = M
    # Fair CRPS from sorted members: sum_ij |xi - xj| = 2 sum_i (2i - M - 1) x_(i).
    w = (2 * np.arange(1, M + 1) - M - 1).reshape(M, 1, 1)

    # One lead at a time keeps float64 working copies to ~10 frames.
    for i in range(L):
        fc, obs = fc_all[:, i].astype(np.float64), obs_all[i].astype(np.float64)
        valid = np.isfinite(obs) & np.isfinite(fc).all(axis=0)
        n = valid.sum()
        if n == 0:               # truth missing for this frame
            continue
        f, o = fc[:, valid], obs[valid]                   # (M, n), (n,)
        pair = 2 * (w[:, :, 0] * np.sort(f, axis=0)).sum(axis=0)
        out["n"][i]    = n
        out["err"][i]  = (f.mean(axis=0) - o).sum()
        out["se"][i]   = ((f.mean(axis=0) - o) ** 2).sum()
        out["var"][i]  = f.var(axis=0, ddof=1).sum()
        out["crps"][i] = (np.abs(f - o).mean(axis=0) - pair / (2 * M * (M - 1))).sum()
        out["grad_obs"][i] = _grad(np.where(valid, obs, np.nan)) * n
        out["grad_fc"][i]  = np.mean([_grad(np.where(valid, m, np.nan)) for m in fc]) * n
    return out


def score_dir(root: Path, names: list[str], var: str) -> dict[str, np.ndarray]:
    sums: dict = {}
    for name in names:
        for k, v in score_file(root / name, var).items():
            sums[k] = v if k == "M" else sums.get(k, 0) + v
    n, M = sums["n"], sums["M"]
    rmse   = np.sqrt(sums["se"] / n)
    spread = np.sqrt(sums["var"] / n)
    return {
        "bias": sums["err"] / n, "rmse": rmse, "spread": spread, "ratio": spread * np.sqrt((M + 1) / M) / rmse,
        "crps": sums["crps"] / n, "sharp": sums["grad_fc"] / sums["grad_obs"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dirs", nargs="+", type=Path, help="Baseline first")
    p.add_argument("--var", default="surface_global_irradiance")
    p.add_argument("--leads", default="1-6,7-12,13-18",
                   help="Lead-frame groups to tabulate (1-based, inclusive)")
    args = p.parse_args()

    sets = [{str(f.relative_to(d)) for f in d.rglob("*.nc")} for d in args.dirs]
    names = sorted(set.intersection(*sets))
    if not names:
        raise SystemExit("No forecast file is present in every directory.")
    print(f"{len(names)} common forecasts, variable {args.var}")

    scores = {d.name: score_dir(d, names, args.var) for d in args.dirs}
    groups = [tuple(int(x) for x in g.split("-")) for g in args.leads.split(",")]
    for lo, hi in groups:
        print(f"\nLead frames {lo}-{hi} ({lo * 10}-{hi * 10} min)")
        print(f"{'':<22}" + "".join(f"{m:>10}" for m in METRICS))
        for name, s in scores.items():
            # Group means of the per-lead values; leads carry equal pixel counts.
            row = [np.mean(s[m][lo - 1:hi]) for m in METRICS]
            print(f"{name:<22}" + "".join(f"{v:>10.3f}" for v in row))


if __name__ == "__main__":
    main()
