import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path
from dask.distributed import Client
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys, os
import json

from metpy.calc import dewpoint_from_specific_humidity
from metpy.units import units

import xesmf as xe


dit_dataset = sys.argv[1]

ds_to_year = {
    "train": 2020,
    "val": 2021,
    "test": 2022,
}

year = ds_to_year[dit_dataset]

##################################################################################
# Fixed parameters
##################################################################################
lat_min=-35
lat_max=-28.5
lon_min=145
lon_max=151.5

# VARS FROM HIMAWARI HELIOSAT
helio_vars = [
    'surface_global_irradiance',
    'cloud_optical_depth',
    'solar_elevation',
]

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

#################################################################################
# Functions to load and process data
#################################################################################

def get_heliosat(date, variables, lat_min, lat_max, lon_min, lon_max, patch_size=256):
    '''
    Use xarray's open_mfdataset() to open Himawari heliosat netcdf files, using arguments optimised for
    opening climate datasets quickly and efficiently.

    INPUTS
    date (str): in format YYYY-MM, year and month to get data for
    variables (list): variables in file to keep
    lat_min, lat_max, lon_min, lon_max (int): region boundaries (must be larger enough to fit patch_size ** 2)
    patch_size (int): multiple of 8, slices into square for torch.nn to handle cleanly

    OUTPUT
    xarray dataset with data_vars=variables and lat/lon dimensions of size=patch_size taken from within region boundaries
    '''

    year, month = date.split("-")
    file_path = Path(f'/g/data/rv74/satellite-products/arc/der/himawari-ahi/solar/p1s/v1.1/{year}/{month}/')
    files = sorted([f for f in file_path.rglob("*.nc")])

    
    def preprocess(ds):
        return ds.sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max)
        )[helio_vars].isel(
            latitude=slice(0, patch_size),
            longitude=slice(-patch_size, None)
        )
    

    return xr.open_mfdataset(
            files,
            preprocess = preprocess,
            concat_dim='time',
            combine='nested',
            data_vars='minimal',
            coords='minimal',
            compat='override',
            parallel=True,
            # chunks={'time':100, 'latitude':-1, 'longitude':-1}
            chunks='auto'
        )

def get_barra(date, std_vars, conv_vars, lat_min, lat_max, lon_min, lon_max):

    '''
    Use xarray's open_mfdataset() to open BARRA-R2 netcdf files, and calculate additional convective parameters

    INPUTS
    date (str): in format YYYY-MM, year and month to get data for
    std_vars, conv_vars (list): variables to retrieve
    lat_min, lat_max, lon_min, lon_max (int): region boundaries (must be larger enough to fit patch_size ** 2)

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
        chunks={'time':10_000, 'lat':-1, 'lon':-1}
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


def get_valid_start_times(ds, context_length, forecast_length):
    '''
    Find all times where the full range of data needed is available, i.e. the full context lenght and forecast length

    INPUTS
    ds: dataset to find valid times for
    context_legnth: number of timesteps the model uses to make the forecast
    forecast_length: number of timesteps forecasted

    OUTPUTS
    xxxxxx
    '''
    total_length    = context_length + forecast_length
    # Build a pandas DatetimeIndex from the helio time coordinate.        #
    # This is a cheap .values call — no Dask compute needed.              #
    times = pd.DatetimeIndex(ds.time.values)
    
    # Compute the gap between each consecutive pair of timestamps.        #
    gaps = times.to_series().diff().fillna(pd.Timedelta("999h"))
    
    # A frame is "continuous" if the gap from the previous frame is exactly 10 minutes
    is_continuous = (gaps == timestep)
    
    # For each position i, we need the (total_length - 1) frames that follow it to ALL be continuous
    continuous_series = is_continuous.astype(int)
    
    # Rolling min over the NEXT (total_length - 1) frames:
    # Reverse → rolling min over trailing window → reverse back.
    # This gives, at position i, the minimum continuity of frames i+1..i+N-1.
    rolling_min = (
        continuous_series
        .iloc[::-1]                                    # reverse
        .rolling(window=total_length - 1, min_periods=total_length - 1)
        .min()
        .iloc[::-1]                                    # reverse back
        .shift(-(total_length - 2))                    # align: result at i covers i+1..i+N-1
    )
    
    # A valid start frame is one where all subsequent frames are continuous
    valid_mask = rolling_min == 1.0
    
    # Also ensure the sequence doesn't run off the end of the array
    valid_mask.iloc[-(total_length - 1):] = False
    
    return times[valid_mask.values]

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
    helio_stats = {}
    bar_stats = {}
    all_valid_times = []

    # to find valid times
    context_length  = 12   # frames
    forecast_length = 6    # frames
    timestep        = pd.Timedelta("10min")

    base_data_dir = Path("/scratch/er8/cd3022/CPDiT/DiT_data/")
    helio_data_dir = base_data_dir / "heliosat" / dit_dataset
    bar_data_dir = base_data_dir / "barra" / dit_dataset
    os.makedirs(helio_data_dir, exist_ok=True)
    os.makedirs(bar_data_dir, exist_ok=True)

    for month in range(1, 13):
        
        date = f"{year}-{month:02d}"

        # load himawari heliosat data
        helio = get_heliosat(
            date,
            variables=helio_vars,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max
        )
        # himawari has some data from the previous UTC day, because of the AEST day. This aligns it with BARRA
        helio = helio.sel(time=slice(f"{date}-01", None))
        # chunk over time
        helio = helio.chunk({'time':300, 'latitude':-1, 'longitude':-1})
        # fill missing timestep
        helio = interp_himawari_gaps(helio) 
        # record the valid times
        all_valid_times.append(get_valid_start_times(helio, context_length, forecast_length))

        
        # now that heliosat data is 256x256, use that as target grid
        if month == 1: 
            lat_min, lat_max = helio.latitude.min().item(), helio.latitude.max().item()
            lon_min, lon_max = helio.longitude.min().item(), helio.longitude.max().item()
        # load BARRA-R2 data
        bar = get_barra(date, std_vars, conv_vars, lat_min, lat_max, lon_min, lon_max)

        bar = bar[['KI', 'hus500']] # start with small subset of vars



        # save monthly netcdf files for each dataset
        helio_file_name = helio_data_dir / f"heliosat_{date}.nc"
        helio.to_netcdf(helio_file_name)
        bar_file_name = bar_data_dir / f"barra_{date}.nc"
        bar.to_netcdf(bar_file_name)

        

        # calculate stats for the month
        for var in helio.data_vars:
            mean = helio[var].mean().compute().item()
            std = helio[var].std().compute().item()

            if month == 1:
                helio_stats[var] = {
                    "mean": [mean],
                    "std": [std],
                }
            else:
                helio_stats[var]["mean"].append(mean)
                helio_stats[var]["std"].append(std)
        
        for var in bar.data_vars:
            mean = bar[var].mean().compute().item()
            std = bar[var].std().compute().item()

            if month == 1:
                bar_stats[var] = {
                    "mean": [mean],
                    "std": [std],
                }
            else:
                bar_stats[var]["mean"].append(mean)
                bar_stats[var]["std"].append(std)


    #######################################################################
    # Save the valid times to a parquet file for later use
    #######################################################################
    all_valid_times = pd.DatetimeIndex(np.concatenate(all_valid_times))

    total_length = context_length + forecast_length

    # save parquet file with valid times
    index_df = pd.DataFrame({
        "start_time":    all_valid_times,
        "context_end":   all_valid_times + (context_length - 1) * timestep,
        "forecast_end":  all_valid_times + (total_length   - 1) * timestep,
    })

    index_dir = Path("/scratch/er8/cd3022/CPDiT/index/")
    os.makedirs(index_dir, exist_ok=True)
    index_df.to_parquet(index_dir / f"{dit_dataset}_index.parquet", index=False)

    #######################################################################
    # Save mean and standard for each variable to a json file for later use
    #######################################################################

    # Take the mean across months (ignoring the different numbers of days) to get final stats for the whole year
    helio_stats_final = helio_stats.copy()
    for var in helio_stats:
        helio_stats_final[var]["mean"] = np.mean(helio_stats[var]["mean"]).item()
        helio_stats_final[var]["std"] = np.mean(helio_stats[var]["std"]).item()

    bar_stats_final = bar_stats.copy()
    for var in bar_stats:
        bar_stats_final[var]["mean"] = np.mean(bar_stats[var]["mean"]).item()
        bar_stats_final[var]["std"] = np.mean(bar_stats[var]["std"]).item()

    # write out stats to a json file for each dataset
    stats_dir = Path("/scratch/er8/cd3022/CPDiT/stats/")
    os.makedirs(stats_dir, exist_ok=True)

    # TO DO
    # MAKE SEPARATE SCRIPT FOR STATS ACROSS ALL YEARS, SO THAT TRAIN/VAL/TEST HAVE SAME STATS
    helio_stats_file = stats_dir / "heliosat_stats.json"
    bar_stats_file = stats_dir / "barra_stats.json"

    with open(helio_stats_file, "w") as file:
        json.dump(helio_stats_final, file, indent=4)

    with open(bar_stats_file, "w") as file:
        json.dump(bar_stats_final, file, indent=4)

    #######################################################################
    # Save regridding weights for BARRA → Heliosat, to be reused across runs
    #######################################################################
    regridder = xe.Regridder(bar, helio, method='bilinear')
    regridder.to_netcdf('/scratch/er8/cd3022/CPDiT/regridders/barra_to_heliosat.nc')   # reuse across runs

    print("Data preparation complete.")
