import json
import pickle
import numpy as np
import xarray as xr
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
import os

DATA_DIR  = Path("/scratch/er8/cd3022/CPDiT/DiT_data/zarr/")
STATS_DIR = Path("/scratch/er8/cd3022/CPDiT/stats/")

# Variables to apply quantile normalisation to instead of z-score.
# Add any future skewed variables here.
QUANTILE_VARS = {"KI"}
N_QUANTILES   = 10_000

HELIO_FILES = [f for f in DATA_DIR.glob("helio*")]
BARRA_FILES = [f for f in DATA_DIR.glob("barra*")]

helio_ds = xr.open_mfdataset(HELIO_FILES)
barra_ds = xr.open_mfdataset(BARRA_FILES)

os.makedirs(STATS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Heliosat — standard z-score for all variables
# ---------------------------------------------------------------------------
helio_stats = {}

for var in helio_ds.data_vars:
    print(f"Heliosat: computing stats for {var}...")
    mean = float(helio_ds[var].mean().compute().item())
    std  = float(helio_ds[var].std().compute().item())
    helio_stats[var] = {"mean": mean, "std": std, "transform": "none"}
    print(f"  mean={mean:.4f}  std={std:.4f}")

with open(STATS_DIR / "heliosat_stats.json", "w") as f:
    json.dump(helio_stats, f, indent=4)
print(f"\nSaved heliosat_stats.json\n")

# ---------------------------------------------------------------------------
# BARRA — z-score for most variables, quantile transform for skewed ones
# ---------------------------------------------------------------------------
barra_stats      = {}
barra_transforms = {}

for var in barra_ds.data_vars:
    print(f"BARRA: computing stats for {var}...")
    arr = barra_ds[var].values.astype("float32").ravel()
    arr = arr[np.isfinite(arr)]

    if var in QUANTILE_VARS:
        print(f"  Applying quantile transform (n_quantiles={N_QUANTILES})...")
        qt = QuantileTransformer(
            n_quantiles         = N_QUANTILES,
            output_distribution = "normal",
            subsample           = min(len(arr), 2_000_000),
            random_state        = 42,
        )
        transformed = qt.fit_transform(arr.reshape(-1, 1)).ravel()
        barra_transforms[var] = qt

        # The quantile transform already produces ~N(0,1), but we store
        # mean/std of the transformed values so the normalisation step
        # in the dataloader is consistent with the heliosat variables.
        mean = float(transformed.mean())
        std  = float(transformed.std())

        norm = (transformed - mean) / (std + 1e-8)
        print(f"  transformed mean={mean:.4f}  std={std:.4f}")
        print(f"  → norm range [{norm.min():.2f}, {norm.max():.2f}]")

        barra_stats[var] = {"mean": mean, "std": std, "transform": "quantile"}

    else:
        mean = float(arr.mean())
        std  = float(arr.std())
        norm = (arr - mean) / (std + 1e-8)
        print(f"  mean={mean:.4f}  std={std:.4f}")
        print(f"  → norm range [{norm.min():.2f}, {norm.max():.2f}]")

        barra_stats[var] = {"mean": mean, "std": std, "transform": "none"}

with open(STATS_DIR / "barra_stats.json", "w") as f:
    json.dump(barra_stats, f, indent=4)
print(f"\nSaved barra_stats.json")

with open(STATS_DIR / "barra_quantile_transforms.pkl", "wb") as f:
    pickle.dump(barra_transforms, f)
print(f"Saved barra_quantile_transforms.pkl")
