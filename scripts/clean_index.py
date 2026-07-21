"""
scripts/clean_index.py

Removes any index entries whose full 18-frame heliosat window contains
a timestamp not present in the actual heliosat files.

Usage:
    python scripts/clean_index.py
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

TOTAL_LENGTH    = 18
TIMESTEP        = pd.Timedelta("10min")
HELIOSAT_DIR    = Path("/scratch/er8/cd3022/CPDiT/DiT_data/heliosat")
INDEX_DIR       = Path("/scratch/er8/cd3022/CPDiT/index")

for split in ("train", "val", "test"):
    index_path  = INDEX_DIR / f"{split}_index.parquet"
    helio_dir   = HELIOSAT_DIR / split

    if not index_path.exists():
        print(f"[{split}] index not found, skipping")
        continue

    # Load the index
    index = pd.read_parquet(index_path)
    print(f"[{split}] loaded {len(index)} entries")

    # Build a set of all timestamps actually present in the heliosat files
    helio_files = sorted(helio_dir.glob("heliosat_*.nc"))
    if not helio_files:
        print(f"[{split}] no heliosat files found in {helio_dir}, skipping")
        continue

    ds = xr.open_mfdataset(helio_files, combine="by_coords")
    helio_times = set(ds.time.values)
    ds.close()
    print(f"[{split}] {len(helio_times)} heliosat timestamps available")

    # Check every index entry
    def is_valid(start):
        times = pd.date_range(start=start, periods=TOTAL_LENGTH, freq=TIMESTEP)
        return all(np.datetime64(t) in helio_times for t in times)

    valid_mask  = index["start_time"].apply(is_valid)
    index_clean = index[valid_mask].reset_index(drop=True)

    n_dropped = len(index) - len(index_clean)
    print(f"[{split}] dropped {n_dropped} invalid entries, kept {len(index_clean)}")

    # Overwrite in place
    index_clean.to_parquet(index_path, index=False)
    print(f"[{split}] saved cleaned index to {index_path}")
