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

patch_size = 256

with open(f"/home/548/cd3022/repos/CPDiT/configs/train_config.yaml") as f:
    config = yaml.safe_load(f)

# VARS FROM HIMAWARI HELIOSAT
helio_vars = config["data"]["heliosat_vars"]

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
        n_workers=24,
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
    zarr_dir = base_data_dir / "zarr"
    os.makedirs(zarr_dir, exist_ok=True)

    valid_times = []
    helio_list = []
    barra_list = []
    print("Starting month loop")
    for month in range(1, 3):
        date = f"{year}-{month:02d}"
        print(f"Processing {date}")
    
        # --------------------------------------------------------------------------- #
        # load himawari heliosat data
        # --------------------------------------------------------------------------- #
        # To properly interpolate BARRA grid and times, a larger himawari area is first taken.
        # Then, once data has been interpolated, the edges are trimmed off both to get 256x256
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
        
        # Now that BARRA is regridded, trim edges to get to 256x256 patch
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

        # add month to list for concatenation later
        helio_list.append(helio)
        barra_list.append(bar_regrid)
        print(f"finished month: {month:02d}")

    full_helio = xr.concat(helio_list, dim='time')
    full_barra = xr.concat(barra_list, dim='time')

    # Combine into one dataset
    final_ds = xr.merge([full_helio, full_barra])
    
    # Force Dask to reconcile the chunk graph from the concat seams
    # TO DO:
    # REPLACE HARDCODED 256 WITH REFERENCE TO PATCH SIZE IN YAML CONFIG FILE
    for var in final_ds.data_vars:
        final_ds[var].data = da.rechunk(final_ds[var].data, chunks=(1, 256, 256))
    
    # Ensure correct encoding to speed up read time
    encoding = {
        var: {
            "chunks": (1, 256, 256),
            "compressor": zarr.Blosc(cname="lz4", clevel=1, shuffle=zarr.Blosc.SHUFFLE),
        }
        for var in final_ds.data_vars
    }
    encoding["time"] = {"chunks": (1,)}
    
    # Save monthly combined dataset
    file_name = zarr_dir / f"combined_{split}.zarr"
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
    
    index_dir = Path("/scratch/er8/cd3022/CPDiT/index/")
    os.makedirs(index_dir, exist_ok=True)
    index_df.to_parquet(index_dir / f"{split}_index.parquet", index=False)
    
    print("Data preparation complete.")

