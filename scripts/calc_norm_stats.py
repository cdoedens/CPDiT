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

FILES = [f for f in DATA_DIR.glob("*.zarr*")]

ds = xr.open_mfdataset(FILES, engine="zarr", concat_dim="time", combine="nested")

os.makedirs(STATS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Heliosat — standard z-score for all variables
# ---------------------------------------------------------------------------
stats = {}
transforms = {}

for var in ds.data_vars:
    
    if var in QUANTILE_VARS:
        print(f"  Applying quantile transform (n_quantiles={N_QUANTILES})...")
        arr = ds[var].values.astype("float32").ravel()
        arr = arr[np.isfinite(arr)]
        qt = QuantileTransformer(
            n_quantiles         = N_QUANTILES,
            output_distribution = "normal",
            subsample           = min(len(arr), 2_000_000),
            random_state        = 42,
        )
        transformed = qt.fit_transform(arr.reshape(-1, 1)).ravel()
        transforms[var] = qt

        # The quantile transform already produces ~N(0,1), but we store
        # mean/std of the transformed values so the normalisation step
        # in the dataloader is consistent with the heliosat variables.
        mean = float(transformed.mean())
        std  = float(transformed.std())

        norm = (transformed - mean) / (std + 1e-8)
        print(f"  transformed mean={mean:.4f}  std={std:.4f}")
        print(f"  → norm range [{norm.min():.2f}, {norm.max():.2f}]")

        stats[var] = {"mean": mean, "std": std, "transform": "quantile"}

    else:
        print(f"  Computing stats for {var}...")
        mean = float(ds[var].mean().compute().item())
        std  = float(ds[var].std().compute().item())
        stats[var] = {"mean": mean, "std": std, "transform": "none"}

        print(f"  mean={mean:.4f}  std={std:.4f}")

with open(STATS_DIR / "combined_stats.json", "w") as f:
    json.dump(stats, f, indent=4)
print(f"\nSaved stats.json\n")

with open(STATS_DIR / "quantile_transforms.pkl", "wb") as f:
    pickle.dump(transforms, f)
print(f"Saved quantile_transforms.pkl")
