"""
Data loading utilities for satellite solar nowcasting.

Design
------
CPDiTDataset        — main dataset class. Reads the precomputed valid-timestamp
                      index (parquet), opens pre-converted Zarr stores once per
                      worker, normalises all variables using precomputed
                      per-variable statistics, and returns (context, forecast)
                      tensor pairs ready for the model.

                      BARRA data is expected to be pre-regridded to the heliosat
                      grid during data preparation (see scripts/prepare_data.py).

build_dataloader    — convenience factory that constructs a CPDiTDataset and
                      wraps it in a DataLoader with the correct settings.

Zarr stores are opened once per worker inside _ensure_open and cached for the
lifetime of that worker process, giving O(1) per-sample time-slice access with
no file-open overhead.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Worker initialisation
# ---------------------------------------------------------------------------

def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker independently."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class CPDiTDataset(Dataset):
    """
    Dataset for the CPDiT latent diffusion model.

    Each sample is identified by a start timestamp drawn from the
    precomputed valid-timestamp index. On __getitem__ the dataset:

      1. Selects the time window from the cached Zarr stores (no file open).
      2. Normalises every variable to zero mean / unit variance.
         Variables with "transform": "quantile" in barra_stats.json are
         passed through the fitted QuantileTransformer before z-scoring.
      3. Stacks all variables along the channel axis and returns
         (context, forecast) tensors of shape (T, C, H, W) each.

    Zarr stores are opened once per worker on the first __getitem__ call,
    then cached for the lifetime of the worker.

    BARRA data must be pre-regridded to the heliosat grid before training.
    See scripts/prepare_data.py.
    """

    def __init__(
        self,
        index_path:                    str | Path,
        heliosat_zarr_path:            str | Path,
        barra_zarr_path:               str | Path,
        heliosat_stats_path:           str | Path,
        barra_stats_path:              str | Path,
        heliosat_vars:                 List[str],
        barra_vars:                    List[str],
        context_length:                int = 12,
        forecast_length:               int = 6,
        satellite_timestep:            str = "10min",
        barra_quantile_transform_path: str | Path | None = None,
    ):
        self.heliosat_zarr_path  = Path(heliosat_zarr_path)
        self.barra_zarr_path     = Path(barra_zarr_path)
        self.heliosat_stats_path = Path(heliosat_stats_path)
        self.barra_stats_path    = Path(barra_stats_path)
        self.heliosat_vars       = heliosat_vars
        self.barra_vars          = barra_vars
        self.context_length      = context_length
        self.forecast_length     = forecast_length
        self.total_length        = context_length + forecast_length
        self.satellite_timestep  = pd.Timedelta(satellite_timestep)

        for p in (self.heliosat_zarr_path, self.barra_zarr_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Zarr store not found: {p}\n"
                    f"Run scripts/prepare_data.py first."
                )

        index = pd.read_parquet(index_path)
        self.start_times = pd.DatetimeIndex(index["start_time"].values)

        with open(self.heliosat_stats_path) as f:
            self._helio_stats: Dict[str, Dict] = json.load(f)
        with open(self.barra_stats_path) as f:
            self._barra_stats: Dict[str, Dict] = json.load(f)

        self._barra_quantile_transforms: Dict = {}
        if barra_quantile_transform_path is not None:
            qt_path = Path(barra_quantile_transform_path)
            if not qt_path.exists():
                raise FileNotFoundError(
                    f"Quantile transform file not found: {qt_path}\n"
                    f"Run scripts/fit_barra_quantile_transform.py first."
                )
            with open(qt_path, "rb") as f:
                self._barra_quantile_transforms = pickle.load(f)

        for var in self.barra_vars:
            s = self._barra_stats.get(var, {})
            if s.get("transform") == "quantile" and var not in self._barra_quantile_transforms:
                import warnings
                warnings.warn(
                    f"Variable '{var}' has transform='quantile' in barra_stats.json "
                    f"but no quantile transform was loaded. Falling back to z-score only. "
                    f"Pass barra_quantile_transform_path to CPDiTDataset to fix this.",
                    UserWarning,
                    stacklevel=2,
                )

        self._helio_ds = None
        self._barra_ds = None

    # ------------------------------------------------------------------ #
    # Lazy asset initialisation (once per worker)                         #
    # ------------------------------------------------------------------ #

    def _ensure_open(self) -> None:
        if self._helio_ds is not None:
            return

        self._helio_ds = zarr.open(str(self.heliosat_zarr_path), mode="r")
        self._barra_ds = zarr.open(str(self.barra_zarr_path),    mode="r")

        DATA_EPOCH = pd.Timestamp("2000-01-01")

        helio_times = DATA_EPOCH + pd.to_timedelta(
            self._helio_ds["time"][:].astype(np.int64), unit="s"
        )
        barra_times = DATA_EPOCH + pd.to_timedelta(
            self._barra_ds["time"][:].astype(np.int64), unit="s"
        )

        self._helio_time_to_idx = {t: i for i, t in enumerate(helio_times)}
        self._barra_time_to_idx = {t: i for i, t in enumerate(barra_times)}

    # ------------------------------------------------------------------ #
    # Normalisation                                                        #
    # ------------------------------------------------------------------ #

    def _normalise_helio(self, arr: np.ndarray, var: str) -> np.ndarray:
        s = self._helio_stats[var]
        return ((arr - s["mean"]) / (s["std"] + 1e-8)).astype(np.float32)

    def _normalise_barra(self, arr: np.ndarray, var: str) -> np.ndarray:
        s   = self._barra_stats[var]
        out = arr.copy()
        if s.get("transform") == "quantile" and var in self._barra_quantile_transforms:
            qt    = self._barra_quantile_transforms[var]
            shape = out.shape
            out   = qt.transform(out.ravel().reshape(-1, 1)).ravel().reshape(shape)
        return ((out - s["mean"]) / (s["std"] + 1e-8)).astype(np.float32)

    def denormalise_barra(self, arr: np.ndarray, var: str) -> np.ndarray:
        s   = self._barra_stats[var]
        out = (arr * (s["std"] + 1e-8) + s["mean"]).astype(np.float32)
        if s.get("transform") == "quantile" and var in self._barra_quantile_transforms:
            qt    = self._barra_quantile_transforms[var]
            shape = out.shape
            out   = qt.inverse_transform(
                out.ravel().reshape(-1, 1)
            ).ravel().reshape(shape).astype(np.float32)
        return out

    def denormalise_helio(self, arr: np.ndarray, var: str) -> np.ndarray:
        s = self._helio_stats[var]
        return (arr * (s["std"] + 1e-8) + s["mean"]).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Dataset protocol                                                     #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.start_times)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
    
        start      = self.start_times[idx]
        end_offset = self.total_length - 1
        end        = start + end_offset * self.satellite_timestep
    
        # --- Heliosat: direct integer slice, no dask ---
        i0_h = self._helio_time_to_idx[start]
        i1_h = i0_h + self.total_length
        helio_arrays = [
            self._helio_ds[var][i0_h:i1_h].astype(np.float32)
            for var in self.heliosat_vars
        ]
    
        # --- BARRA: nearest-hour lookup then direct slice ---
        barra_start = start.round("h")
        i0_b = self._barra_time_to_idx.get(barra_start)
        if i0_b is None:
            # fall back to closest available
            i0_b = min(self._barra_time_to_idx,
                       key=lambda t: abs(t - start))
            i0_b = self._barra_time_to_idx[i0_b]
        i1_b = i0_b + self.total_length
        barra_arrays = [
            self._barra_ds[var][i0_b:i1_b].astype(np.float32)
            for var in self.barra_vars
        ]
    
        # --- Normalise and stack ---
        channel_arrays = []
        for var, arr in zip(self.heliosat_vars, helio_arrays):
            channel_arrays.append(self._normalise_helio(arr, var))
        for var, arr in zip(self.barra_vars, barra_arrays):
            channel_arrays.append(self._normalise_barra(arr, var))
    
        data = np.stack(channel_arrays, axis=1)  # (T, C, H, W)
    
        assert data.shape[0] == self.total_length, (
            f"Expected {self.total_length} timesteps, got {data.shape[0]} "
            f"for start_time={start}."
        )
    
        data_tensor = torch.from_numpy(data)
        context     = data_tensor[: self.context_length]
        forecast    = data_tensor[self.context_length :]
        return context, forecast

    def __repr__(self) -> str:
        return (
            f"CPDiTDataset("
            f"n_samples={len(self)}, "
            f"context_length={self.context_length}, "
            f"forecast_length={self.forecast_length}, "
            f"heliosat_vars={self.heliosat_vars}, "
            f"barra_vars={self.barra_vars})"
        )


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------

class SatelliteDataset(CPDiTDataset):
    """Backward-compatible alias for the main dataset."""


class HIMAWARIDataset(CPDiTDataset):
    """Backward-compatible alias for the main dataset."""


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    split:   str,
    config:  dict,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    if split not in ("train", "val", "test"):
        raise ValueError(
            f"split must be 'train', 'val', or 'test', got '{split}'"
        )

    data_cfg = config["data"]
    if shuffle is None:
        shuffle = (split == "train")

    dataset = CPDiTDataset(
        index_path                    = data_cfg["valid_timestamps"][split],
        heliosat_zarr_path            = data_cfg["heliosat_zarr"][split],
        barra_zarr_path               = data_cfg["barra_zarr"][split],
        heliosat_stats_path           = data_cfg["normalisation_stats"]["heliosat"],
        barra_stats_path              = data_cfg["normalisation_stats"]["barra"],
        heliosat_vars                 = data_cfg["heliosat_vars"],
        barra_vars                    = data_cfg["barra_vars"],
        context_length                = data_cfg["context_length"],
        forecast_length               = data_cfg["forecast_length"],
        satellite_timestep            = f"{data_cfg['satellite_timestep_min']}min",
        barra_quantile_transform_path = data_cfg["normalisation_stats"].get(
                                            "barra_quantile_transforms"
                                        ),
    )

    return DataLoader(
        dataset,
        batch_size         = config["training"]["batch_size"][f"stage{config['training']['stage']}"],
        shuffle            = shuffle,
        num_workers        = data_cfg["num_workers"],
        pin_memory         = data_cfg.get("pin_memory", True) and data_cfg["num_workers"] > 0,
        prefetch_factor    = data_cfg.get("prefetch_factor", 2) if data_cfg["num_workers"] > 0 else None,
        worker_init_fn     = _worker_init_fn,
        persistent_workers = data_cfg["num_workers"] > 0,
    )
