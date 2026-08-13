import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys, os
import json
import yaml
import shutil

from metpy.calc import dewpoint_from_specific_humidity
from metpy.units import units

from dask.distributed import Client
import dask.array as da
import zarr
from numcodecs import LZ4

import pvlib

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
sys.path.append("/home/548/cd3022/repos/CPDiT/")
import src.prep as prep

split = sys.argv[1]

ds_to_year = {
    "train": 2020,
    "val": 2021,
    "test": 2022,
}
year = ds_to_year[split]

##################################################################################
# Fixed parameters
##################################################################################
lat_min=-35
lat_max=-28.5
lon_min=145
lon_max=151.5

with open(f"/home/548/cd3022/repos/CPDiT/configs/train_config.yaml") as f:
    config = yaml.safe_load(f)

# size of the image patches to extract from the data
patch_size = config["model"]["image_size"]

# STANDARD BARRA VARS
std_vars = [
    # Moisture
    'huss',
    'hus850',
    'hus700',
    'hus500',
    # Temperature
    'tas',
    'ta850',
    'ta700',
    'ta500',
    # radiation
    # 'rsds',
]


# CONVECTIVE BARRA VARS
conv_vars = [
    'RH24mean',
    'MUEL',
    'FZL',
    'MULCL'
]

# BARRA VARS FOR MODEL
# (separate from above, to allow for vars like KI to be calculated)
barra_vars = config["data"]["barra_vars"]
# VARS FROM HIMAWARI HELIOSAT
helio_vars = config["data"]["heliosat_vars"]

all_vars = list(set(barra_vars + helio_vars))

# TIMESTEPS NEEDED
context_length  = config["data"]["context_length"]
forecast_length = config["data"]["forecast_length"]
total_length = context_length + forecast_length

# USE TO FILTER TIMESTEPS WITH LOW SOLAR ELEVATION
syd_lat = -31.75
syd_lon = 148.25

all_times = pd.date_range(
    start=f"{year-1}-12-31 00:00",
    end=f"{year}-12-31 23:50",
    freq="10min"
)

solar_elevation = pvlib.solarposition.get_solarposition(
    all_times, syd_lat, syd_lon
).elevation

daytime_times = set(solar_elevation[solar_elevation >= 10.0].index)

if __name__ == "__main__":
    client = Client(
        n_workers=48,
        threads_per_worker=1
    )

    ################################################################################
    # 
    # START DATA PROCESSING
    #
    ################################################################################
    
    # to find valid times
    timestep = pd.Timedelta("10min")
    
    base_data_dir = Path("/scratch/er8/cd3022/CPDiT/DiT_data/")
    save_dir = base_data_dir / "testing"
    os.makedirs(save_dir, exist_ok=True)

    valid_times = []
    helio_list = []
    barra_list = []
    print("Starting month loop")
    for month in range(1, 13):
        date = f"{year}-{month:02d}"
        print(f"Processing {date}")
    
        # --------------------------------------------------------------------------- #
        # load himawari heliosat data
        # --------------------------------------------------------------------------- #
        # To properly interpolate BARRA grid and times, a larger himawari area is first taken.
        # Then, once data has been interpolated, the edges are trimmed off both to get patch_size**2

        # wrapping entire month in try-except to skip if any data is missing
        try:
            helio = prep.get_heliosat(
                date,
                variables=helio_vars,
                lat_min=lat_min-0.5,
                lat_max=lat_max+0.5,
                lon_min=lon_min-0.5,
                lon_max=lon_max+0.5
            )

        
        
            # fill missing timestep
            helio = prep.interp_himawari_gaps(helio) 
            # himawari has some data from the previous UTC day, because of the AEST day. This aligns it with BARRA
            helio = helio.sel(time=slice(f"{date}-01", None))
            
            
            # load BARRA-R2 data
            bar = prep.get_barra(
                date,
                std_vars, conv_vars,
                lat_min=lat_min-0.5,
                lat_max=lat_max+0.5,
                lon_min=lon_min-0.5,
                lon_max=lon_max+0.5
            )
            
            bar = bar[barra_vars] # just the vars for this model config
            
            # Regrid to himawari resolution
            bar_regrid = bar.interp(
                lat=helio.latitude,
                lon=helio.longitude,
                time=helio.time,
                method='nearest'
            )
            
            # Now that BARRA is regridded, trim edges to get to patch_size x patch_size
            helio = helio.sel(
                latitude=slice(lat_min, lat_max),
                longitude=slice(lon_min, lon_max)
            ).isel(
                latitude=slice(0, patch_size),
                longitude=slice(-patch_size, None)
            )
            bar_regrid = bar_regrid.sel(
                latitude=helio.latitude,
                longitude=helio.longitude,
            )
            # Record the valid times
            # Record the valid times
            print("Finding valid start times")
            monthly_valid_times = prep.get_valid_start_times(helio, total_length, daytime_times, timestep)
            valid_times.append(monthly_valid_times)

        except (FileNotFoundError, OSError) as e:
            print(f"WARNING: Skipping {date} - could not load Himawari data: {e}")
            continue  # Skip to the next month if data is missing
        # add month to list for concatenation later
        helio_list.append(helio)
        barra_list.append(bar_regrid)
        print(f"finished month: {month:02d}")

    full_helio = xr.concat(helio_list, dim='time')
    full_barra = xr.concat(barra_list, dim='time')

    # Combine into one dataset
    final_ds = xr.merge([full_helio, full_barra])

    # drop extra coords leftover from BARRA
    coords_to_keep = {"time", "latitude", "longitude"}
    coords_to_drop = [c for c in final_ds.coords if c not in coords_to_keep]
    final_ds = final_ds.drop_vars(coords_to_drop)

    # Rechunk lazily — no compute triggered, just graph restructuring
    final_ds = final_ds.chunk({
        "time": total_length,
        "latitude": patch_size,
        "longitude": patch_size
        })

    
    # Ensure correct encoding to speed up read time
    encoding = {
        var: {
            "chunks": (total_length, patch_size, patch_size),
            "compressor": None,
        }
        for var in all_vars
    }
    encoding["time"] = {"chunks": (total_length,)}
    
    # Save monthly combined dataset
    file_name = save_dir / f"combined_{split}.zarr"
    final_ds.to_zarr(file_name, mode="w", encoding=encoding)
    
    #######################################################################
    # Save the valid times to a parquet file for later use
    #######################################################################
    valid_times = pd.DatetimeIndex(np.concatenate(valid_times))

    # save parquet file with valid times
    index_df = pd.DataFrame({
        "start_time":    valid_times,
        "context_end":   valid_times + (context_length - 1) * timestep,
        "forecast_end":  valid_times + (total_length   - 1) * timestep,
    })
    
    index_dir = Path("/scratch/er8/cd3022/CPDiT/index_testing/")
    os.makedirs(index_dir, exist_ok=True)
    index_df.to_parquet(index_dir / f"{split}_index.parquet", index=False)
    
    print("Data preparation complete.")