import json
import numpy as np
import xarray as xr
from pathlib import Path
import os

DATA_DIR = Path("/scratch/er8/cd3022/CPDiT/DiT_data/zarr/")
STATS_DIR   = Path("/scratch/er8/cd3022/CPDiT/stats/")

HELIO_FILES = [f for f in DATA_DIR.glob("helio*")]
BARRA_FILES = [f for f in DATA_DIR.glob("barra*")]

helio_ds = xr.open_mfdataset(HELIO_FILES)
barra_ds = xr.open_mfdataset(BARRA_FILES)

helio_stats = {}
barra_stats = {}

for var in helio_ds.data_vars:
    mean = helio_ds[var].mean().compute().item()
    std = helio_ds[var].std().compute().item()
    helio_stats[var] = {
        "mean": mean,
        "std": std
    }

for var in barra_ds.data_vars:
    mean = barra_ds[var].mean().compute().item()
    std = barra_ds[var].std().compute().item()
    barra_stats[var] = {
        "mean": mean,
        "std": std
    }

# write out stats to a json file for each dataset
stats_dir = Path("/scratch/er8/cd3022/CPDiT/stats/")
os.makedirs(stats_dir, exist_ok=True)

helio_stats_file = stats_dir / "heliosat_stats.json"
barra_stats_file = stats_dir / "barra_stats.json"

with open(helio_stats_file, "w") as file:
    json.dump(helio_stats, file, indent=4)

with open(barra_stats_file, "w") as file:
    json.dump(barra_stats, file, indent=4)