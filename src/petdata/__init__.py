"""Data loading via pyearthtools pipeline for the CPDiT training script."""
import traceback
import functools
from typing import Optional

import torch
import xarray as xr
import numpy as np
from torch.utils.data import DataLoader, IterableDataset

import pyearthtools.data as petdata
import pyearthtools.pipeline as petpipe
my_site = 'site_archive_nci'  # set this to 'site_archive_nci', 'site_archive_jasmin' or 'site_archive_met_office'
import importlib
_ = importlib.import_module(my_site)

import warnings
warnings.filterwarnings("ignore",message="In a future version of xarray the default value for join",category=FutureWarning)
warnings.filterwarnings("ignore",message="In a future version of xarray the default value for compat",category=FutureWarning)
warnings.filterwarnings("ignore",message="Data requested at a higher resolution than available")
warnings.filterwarnings("ignore",message="Importing `spectral_angle_mapper` from `torchmetrics.functional`",category=FutureWarning)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def _build_pipeline(split: str, config: dict) -> petpipe.Pipeline:
    """Build a pyearthtools Pipeline for a given split (train/val)."""

    data_cfg     = config["data"]
    split_cfg    = data_cfg["splits"][split]

    start_date   = split_cfg["start"]
    end_date     = split_cfg["end"]
    sat_timestep = data_cfg.get("sat_timestep", "10 minutes")
    bar_timestep = data_cfg.get("bar_timestep", "1 hour")

    # ── Himawari ──────────────────────────────────────────────────────────
    himawari_vars = data_cfg.get("himawari_vars", ["surface_global_irradiance"])
    himawari = petdata.archive.Himawari(himawari_vars)

    n_prior_sat  = data_cfg.get("n_prior_sat", 12)   # context frames
    n_prior_bar  = n_prior_sat // 6
    n_post       = data_cfg.get("n_post", 1)
    # wide window to account for edges needed for interpolation
    temporal_window_sat = petpipe.modifications.TemporalWindow(
        prior_indexes=[i for i in range(-n_prior_sat - 1, 0)],
        posterior_indexes=[i for i in range(0, n_post + 1)],
        merge_method = functools.partial(xr.concat, dim='time'),
        timedelta=petdata.time.TimeDelta((10, "minutes"))
    )

    ################################################
    # TO DO: UPDATE NORMALISATION
    ################################################
    bounds       = data_cfg.get("bounds", [-35, -28.5, 145, 151.5])
    sat_norm     = data_cfg.get("sat_norm", 1200)

    sat_pipe = petpipe.Pipeline(
        himawari,
        petpipe.operations.xarray.Sort(order=['time', 'latitude', 'longitude']),  #
        # Align the data variable's coordinate order to the dataset coordinate order so all arrays are the same shape
        petpipe.operations.xarray.AlignDataVariableDimensionsToDatasetCoords(),
        petdata.transform.region.Bounding(*bounds),
        petpipe.operations.xarray.normalisation.SingleValueDivision(sat_norm),
        temporal_window_sat,
        petpipe.operations.xarray.Merge(),
        # iterator=petpipe.iterators.DateRange(start_date, end_date, interval=sat_timestep),
        exceptions_to_ignore=petdata.exceptions.DataNotFoundError
    )

    # ── BARRA ─────────────────────────────────────────────────────────────
    barra_vars   = data_cfg.get("barra_vars", ["RH24mean"])
    barra_domain = data_cfg.get("barra_domain", "AUST-11")
    barra_freq   = data_cfg.get("barra_freq", "1hr")
    barra_norm   = data_cfg.get("barra_norm", 100)

    barra_conv = petdata.archive.BARRA_V2(
        barra_vars,
        domain_id  = barra_domain,
        frequency  = barra_freq,
    )

    # wide window to account for edges needed for interpolation, and for BARRA low resolution
    temporal_window_bar = petpipe.modifications.TemporalWindow(
        prior_indexes=[i for i in range(-n_prior_bar - 1, 0)],
        posterior_indexes=[i for i in range(0, n_post + 1)],
        merge_method = functools.partial(xr.concat, dim='time'),
        timedelta=petdata.time.TimeDelta((1, "h"))
    )

    bar_pipe = petpipe.Pipeline(
        barra_conv,
        petdata.transforms.coordinates.Drop("crs"),
        petpipe.operations.xarray.Sort(order=['time', 'latitude', 'longitude']),
        # Align the data variable's coordinate order to the dataset coordinate order so all arrays are the same shape
        petpipe.operations.xarray.AlignDataVariableDimensionsToDatasetCoords(),
        petdata.transform.region.Bounding(*bounds),  # cut down on region for example
        petpipe.operations.xarray.normalisation.SingleValueDivision(100),
        temporal_window_bar,
        petpipe.operations.xarray.Merge(),
        # iterator=petpipe.iterators.DateRange(start_date, end_date, interval=bar_timestep),
        exceptions_to_ignore=petdata.exceptions.DataNotFoundError,
    )

    full_pipe = petpipe.Pipeline(
        (sat_pipe, bar_pipe),
        iterator=petpipe.iterators.DateRange(start_date, end_date, interval=sat_timestep),
        exceptions_to_ignore=petdata.exceptions.DataNotFoundError,
    )

    return full_pipe


# ---------------------------------------------------------------------------
# IterableDataset wrapper
# ---------------------------------------------------------------------------

class PipelineDataset(IterableDataset):
    """
    Wraps a pyearthtools Pipeline as a PyTorch IterableDataset.

    Each sample from the pipeline has shape (t, c, h, w).
    The last time index is treated as the forecast frame;
    all prior indices are the context.

    Yields:
        context:  (t-1, c, h, w)  float32 tensor
        forecast: (1,   c, h, w)  float32 tensor
    """

    def __init__(
        self,
        pipeline: petpipe.Pipeline,
        config: dict
    ):
        super().__init__()
        self.pipeline = pipeline
        self.config   = config
        
    def __iter__(self):
        image_size = self.config["model"].get("image_size", 256)
        pipeline_iter = iter(self.pipeline)
        while True:
            try:
                sat_sample, bar_sample = next(pipeline_iter)
            except StopIteration:
                return
            except petdata.exceptions.DataNotFoundError as e:
                print(f"Skipping missing data: {e}")
                continue
            except Exception as e:
                print(f"Skipping sample (fetch error): {type(e).__name__}: {e}")
                traceback.print_exc()  # ← full traceback, not just the message
                continue

            print(f"timestamp: {sat_sample.time.values[-1]}")
             # Interpolate BARRA to Himawari spatial/temporal resolution
            bar_interp = bar_sample.interp(
                latitude=sat_sample.latitude,
                longitude=sat_sample.longitude,
                time=sat_sample.time,
                method='nearest'
            )
            # Combine into one dataset
            combined_sample = xr.merge([sat_sample, bar_interp])
            # Trim the edges to remove NANs from interpolation
            combined_inner = combined_sample.isel(
                time=slice(1, -1),
                latitude=slice(10, 10+image_size),
                longitude=slice(10, 10+image_size),
            )
            arr = np.stack([combined_inner[v].values for v in combined_inner.data_vars], axis=0)
            arr = np.transpose(arr, (1, 0, 2, 3))
            tensor = torch.tensor(arr, dtype=torch.float32)

            if torch.any(torch.isnan(tensor)):
                print("NANs found in tensor sample, skipping...")
                continue

            context  = tensor[:-1]
            forecast = tensor[[-1]]
            yield context, forecast


# ---------------------------------------------------------------------------
# Public entry point called by Trainer.setup_data()
# ---------------------------------------------------------------------------

def build_dataloader(
    split:   str,
    config:  dict,
    shuffle: bool = False,
) -> DataLoader:
    """
    Build a DataLoader for the given split using the pyearthtools pipeline.

    Args:
        split:   "train" or "val"
        config:  full config dict from load_config()
        shuffle: ignored for IterableDataset (pipeline order is date-ordered)

    Returns:
        A DataLoader yielding (context, forecast) tensor pairs.
    """
    pipeline = _build_pipeline(split, config)
    dataset  = PipelineDataset(pipeline, config)

    loader_cfg = config.get("dataloader", {})

    return DataLoader(
        dataset,
        batch_size = loader_cfg.get("batch_size", 8),
        num_workers= 0,          # must be 0 — pipeline is not fork-safe
        pin_memory = loader_cfg.get("pin_memory", True),
    )
