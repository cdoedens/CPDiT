import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys, os
import json
import yaml
import xesmf as xe
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

#################################################################################
# Functions to load and process data
#################################################################################

def get_heliosat(date, variables, lat_min, lat_max, lon_min, lon_max):
    '''
    Use xarray's open_mfdataset() to open Himawari heliosat netcdf files, using arguments optimised for
    opening climate datasets quickly and efficiently.

    INPUTS
    date (str): in format YYYY-MM, year and month to get data for
    variables (list): variables in file to keep
    lat_min, lat_max, lon_min, lon_max (int): region boundaries

    OUTPUT
    xarray dataset with data_vars=variables and lat/lon dimensions taken from within region boundaries
    '''

        
    date_dt = datetime.strptime(date, '%Y-%m')
    if date_dt <= datetime.strptime('2019-03-31', '%Y-%m-%d'):
        version = 'v1.0'
    else:
        version = 'v1.1'

    year, month = date.split("-")
    file_path = Path(f'/g/data/rv74/satellite-products/arc/der/himawari-ahi/solar/p1s/{version}/{year}/{month}/')
    files = sorted([f for f in file_path.rglob("*.nc")])

    
    def preprocess(ds):
        return ds.sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max)
        )[variables]
    

    return xr.open_mfdataset(
            files,
            preprocess = preprocess,
            concat_dim='time',
            combine='nested',
            data_vars='minimal',
            coords='minimal',
            compat='override',
            parallel=True,
            chunks='auto'
        )

def get_barra(date, std_vars, conv_vars, lat_min, lat_max, lon_min, lon_max):

    '''
    Use xarray's open_mfdataset() to open BARRA-R2 netcdf files, and calculate additional convective parameters

    INPUTS
    date (str): in format YYYY-MM, year and month to get data for
    std_vars, conv_vars (list): variables to retrieve
    lat_min, lat_max, lon_min, lon_max (int): region boundaries

    OUTPUT
    xarray dataset with data_vars=[std_vars + conv_vars + dew_points + KI + TCD]  taken from within region boundaries
    '''
    
    year, month = date.split("-")

    files = []
    for var in std_vars:
        file_path = Path(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var}/latest/')
        var_file = [f for f in file_path.glob(f'*{year}{month}.nc')][0]
        files.append(var_file)

    for var in conv_vars:
        file_path = Path(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var}/latest/')
        var_file = [f for f in file_path.glob(f'*{year}{month}.nc')][0]
        files.append(var_file)

    def preprocess(ds):
        return ds.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max),
        )
    files=sorted(files)

    bar =  xr.open_mfdataset(
        files,
        preprocess = preprocess,
        compat='override',
        parallel=True,
        chunks="auto"
    )


    #################################################################################
    # Calculate additional convective indices
    #################################################################################
    # When there is 0 CAPE, MUEL is nan.
    # To fix this and make sure the model is trained off all environments,
    # set MUEL to 0 where there are nan values.
    # Other variables (e.g. CIN) are not so easily set to 0
    bar['MUEL'] = xr.where(bar['MUEL'].isnull(), 0, bar['MUEL'])
    
    
    # Calculate dew points for thunderstorm parameters
    for pressure in ['850', '700', '500']:
        bar[f'dp{pressure}'] = (
            dewpoint_from_specific_humidity(
                pressure=int(pressure) * units.hPa,
                specific_humidity=bar[f'hus{pressure}'] * units('g/g'),
            )
            .metpy.convert_units('K')   # or 'K' depending on your preference
            .metpy.dequantify()            # removes units → returns plain DataArray
        )
    
    # Convective Parameters from RAW TS Climatology Paper
    bar['KI'] = bar['ta850'] - bar['ta500'] + bar['dp850'] - (bar['ta700'] - bar['dp700'])
    # bar['TCD'] = bar['MUEL'] - bar['MULCL']

    return bar


def interp_himawari_gaps(ds):
    '''
    Himawari misses one timestep each day at T02:40.
    This function fills that value with a linear interpolation between adjacent times

    INPUTS
    ds: himawari dataset

    OUTPUTS
    The same dataset but with the missing timestep filled
    '''
    ds_filled = ds.copy()
    gap_mask = (
        (ds.time.dt.hour == 2) &
        (ds.time.dt.minute == 40)
    )
    
    for var in ds.data_vars:
        mask = gap_mask & ds[var].isnull()
        estimate = (
            ds[var].shift(time=1)
            + ds[var].shift(time=-1)
        ) / 2
    
        ds_filled[var] = ds[var].where(~mask, estimate)
    return ds_filled


def get_valid_start_times(ds, total_length, daytime_times, timestep):
    times        = pd.DatetimeIndex(ds.time.values)

    # ------------------------------------------------------------------ #
    # Step 1: Find bad days using only daytime timesteps                  #
    # Spatial mean reduces (T, H, W) → (T,) before resampling,           #
    # so the .compute() only pulls a tiny array.                          #
    # ------------------------------------------------------------------ #

    timestep_has_nan = (
        ds["surface_global_irradiance"]
        .isnull()
        .any(dim=["latitude", "longitude"])   # (T,) bool — True if any pixel NaN
        .compute()
    )
    
    bad_timesteps = set(
        pd.DatetimeIndex(timestep_has_nan.time.values[timestep_has_nan.values])
    )

    print(f"  {len(bad_timesteps)} bad timesteps identified.")

    # ------------------------------------------------------------------ #
    # Step 2: Continuity filter                                          #
    # Finds candidate start times where there are total_length
    # time steps all of 10 mins ahead of the start time, stops window 
    # from going overnight
    # ------------------------------------------------------------------ #
    gaps              = times.to_series().diff().fillna(pd.Timedelta("999h"))
    is_continuous     = (gaps == timestep)
    continuous_series = is_continuous.astype(int)

    rolling_min = (
        continuous_series
        .rolling(window=total_length - 1, min_periods=total_length - 1)
        .min()
        .shift(-(total_length - 1))
    )

    valid_mask = rolling_min == 1.0
    valid_mask.iloc[-(total_length - 1):] = False
    candidate_times = times[valid_mask.values]

    # ------------------------------------------------------------------ #
    # Step 3: Apply both filters — elevation first, then bad days         #
    # Both checks are pure set lookups, no I/O                           #
    # ------------------------------------------------------------------ #
    valid_start_times = [
        t0 for t0 in candidate_times
        if all(
            t0 + i * timestep in daytime_times # check that all  times in the wind have sun in the sky
            for i in range(total_length)
        )
        and not any(
            (t0 + i * timestep) in bad_timesteps # check that there are now nan timesteps in the window
            for i in range(total_length)
        )
    ]

    print(f"  {len(candidate_times)} candidates → "
          f"{len(valid_start_times)} after filter.")

    return pd.DatetimeIndex(valid_start_times)

def get_valid_timesteps(valid_start_times, total_length, timestep):
    """
    Given valid start times, return the union of all timesteps
    that fall within any valid window of length total_length.
    """
    print("finding all valid times, based off valid start times and total_length")
    all_timesteps = set()
    for t0 in valid_start_times:
        for i in range(total_length):
            all_timesteps.add(t0 + i * timestep)
    print("All valid times retrieved")
    return pd.DatetimeIndex(sorted(all_timesteps))