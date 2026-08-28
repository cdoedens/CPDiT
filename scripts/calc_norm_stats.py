# Core Python imports
import os, sys
from pathlib import Path
import json

# Scientific standard imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import functools
sys.path.append("/home/548/cd3022/repos/CPDiT/")
import src.prep as prep

# PyEarthTools imports including NCI cached data
import pyearthtools.data as petdata
import pyearthtools.pipeline as petpipe
from pyearthtools.data.time import Petdt

my_site = 'site_archive_nci'  # set this to 'site_archive_nci', 'site_archive_jasmin' or 'site_archive_met_office'
import importlib
_ = importlib.import_module(my_site)

from dask.distributed import Client

# We specify the date, hour, and minute for querying data
date = '20200105T0000'
start_date = '20200101T0000'
end_date = '20210101T0000'
bar_timestep = "1 hour" 
sat_timestep = "10 min" 
save_dir = Path("/scratch/er8/cd3022/CPDiT/stats/")

sat_mean_path = save_dir / "mean_sat.npy"
sat_std_path = save_dir / "std_sat.npy"
bar_mean_path = save_dir / "mean_bar.npy"
bar_std_path = save_dir / "std_bar.npy"

iterator = petpipe.iterators.DateRange(start_date, end_date, interval=bar_timestep)
iterator_full = petpipe.iterators.DateRange(start_date, end_date, interval=sat_timestep)


if __name__ == "__main__":
    client = Client(
        n_workers=24,
        threads_per_worker=1
    )

    # TO DO: update to full year
    date = "2020-01"

    # TO DO: update to use config file, instead of hard coding
    lat_min, lat_max, lon_min, lon_max = -35, -28.5, 145, 151.5
    him_vars = [
        "surface_global_irradiance",
        "solar_elevation"
    ]
    bar_std = ["tas"]
    bar_conv = [
        "RH24mean"
    ]
    him = prep.get_heliosat(
        date,
        variables=him_vars,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max
    )
    
    # load BARRA-R2 data
    bar = prep.get_barra(
        date,
        bar_std, bar_conv,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max
    )
    
    bar = bar[bar_conv]
    
    stats = {}
    
    for var in him.data_vars:
        mean = him[var].mean().values.item()
        std = him[var].std().values.item()
        stats[var] = {"mean": mean, "std": std}
    for var in bar.data_vars:
        mean = bar[var].mean().values.item()
        std = bar[var].std().values.item()
        stats[var] = {"mean": mean, "std": std}
    
    with open(save_dir / "combined_stats.json", "w") as f:
        json.dump(stats, f, indent=4)