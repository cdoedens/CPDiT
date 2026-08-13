"""
SAME AS SRC.DATA BUT USES TEST DATA INSTEAD
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler


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

      1. Looks up the start timestamp in the unified time index to get
         a contiguous integer slice — O(1), no searching at runtime.
      2. Reads exactly (context_length + forecast_length) timesteps for
         each variable from the combined Zarr store.
      3. Normalises every variable to zero mean / unit variance.
         Variables with "transform": "quantile" in combined_stats.json
         are passed through the fitted QuantileTransformer before z-scoring.
      4. Stacks all variables along the channel axis and returns
         (context, forecast) tensors of shape (T, C, H, W) each.

    The Zarr store is opened once per worker on the first __getitem__ call
    and cached for the lifetime of that worker.
    """

    def __init__(
        self,
        index_path:                    str | Path,
        zarr_path:                     str | Path,
        stats_path:                    str | Path,
        heliosat_vars:                 List[str],
        barra_vars:                    List[str],
        context_length:                int = 12,
        forecast_length:               int = 6,
        satellite_timestep:            str = "10min",
        quantile_transform_path:       str | Path | None = None,
    ):
        """
        Args:
            index_path:              Parquet file of valid start timestamps.
            zarr_path:               Combined Zarr store (heliosat + BARRA,
                                     single shared time axis).
            stats_path:              combined_stats.json — flat dict keyed by
                                     variable name with keys: mean, std, and
                                     optionally transform: "quantile".
            heliosat_vars:           Heliosat variable names to load.
            barra_vars:              BARRA variable names to load.
            context_length:          Number of context timesteps.
            forecast_length:         Number of forecast timesteps.
            satellite_timestep:      Temporal resolution (e.g. "10min").
            quantile_transform_path: Path to quantile_transforms.pkl — a
                                     pickled dict of fitted QuantileTransformers
                                     keyed by variable name.
        """
        self.zarr_path          = Path(zarr_path)
        self.stats_path         = Path(stats_path)
        self.heliosat_vars      = heliosat_vars
        self.barra_vars         = barra_vars
        self.all_vars           = heliosat_vars + barra_vars
        self.context_length     = context_length
        self.forecast_length    = forecast_length
        self.total_length       = context_length + forecast_length
        self.satellite_timestep = pd.Timedelta(satellite_timestep)

        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr store not found: {self.zarr_path}")
        if not self.stats_path.exists():
            raise FileNotFoundError(f"Stats file not found: {self.stats_path}")

        # Load valid start timestamps
        index            = pd.read_parquet(index_path)
        self.start_times = pd.DatetimeIndex(index["start_time"].values)

        # Load normalisation stats
        with open(self.stats_path) as f:
            self._stats: Dict[str, Dict] = json.load(f)

        missing = [v for v in self.all_vars if v not in self._stats]
        if missing:
            raise ValueError(
                f"Variables missing from {self.stats_path.name}: {missing}"
            )

        # Load quantile transforms
        self._quantile_transforms: Dict = {}
        if quantile_transform_path is not None:
            qt_path = Path(quantile_transform_path)
            if not qt_path.exists():
                raise FileNotFoundError(
                    f"Quantile transform file not found: {qt_path}"
                )
            with open(qt_path, "rb") as f:
                self._quantile_transforms = pickle.load(f)

        for var in self.barra_vars:
            if (
                self._stats.get(var, {}).get("transform") == "quantile"
                and var not in self._quantile_transforms
            ):
                warnings.warn(
                    f"Variable '{var}' has transform='quantile' in "
                    f"{self.stats_path.name} but no fitted transformer was "
                    f"found in quantile_transforms.pkl. Falling back to "
                    f"z-score only.",
                    UserWarning,
                    stacklevel=2,
                )

        # Zarr store and time index opened lazily per worker
        self._ds:           zarr.Group | None           = None

    # ------------------------------------------------------------------ #
    # Lazy store initialisation (once per worker)                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _decode_time(raw: np.ndarray, attrs: dict) -> pd.DatetimeIndex:
        """
        Decode a raw integer time array using the CF 'units' attribute,
        e.g. 'minutes since 2020-01-01 00:00:00'.
        """
        units: str = attrs.get("units", "")
    
        # Parse "X since YYYY-MM-DD HH:MM:SS"
        try:
            freq_str, _, origin_str = units.partition(" since ")
            origin   = pd.Timestamp(origin_str.strip())
            freq_map = {
                "minutes": "min",
                "seconds": "s",
                "hours":   "h",
                "days":    "D",
            }
            pd_unit = freq_map.get(freq_str.strip().lower())
            if pd_unit is None:
                raise ValueError(f"Unrecognised time unit: '{freq_str}'")
            times = origin + pd.to_timedelta(raw.astype(np.int64), unit=pd_unit)
        except Exception as e:
            raise RuntimeError(
                f"Could not decode time axis with units='{units}': {e}"
            ) from e
    
        # Strip timezone so lookups against tz-naive index parquet always match
        if times.tz is not None:
            times = times.tz_localize(None)
    
        return times
    
    def _ensure_open(self) -> None:
        if self._ds is not None:
            return
    
        self._ds = zarr.open(str(self.zarr_path), mode="r")
    
        self._time_array = self._decode_time(self._ds["time"][:], dict(self._ds["time"].attrs))
    
        missing_vars = [v for v in self.all_vars if v not in self._ds]
        if missing_vars:
            raise KeyError(
                f"Variables not found in {self.zarr_path.name}: {missing_vars}\n"
                f"Available: {list(self._ds.keys())}"
            )


    # ------------------------------------------------------------------ #
    # Normalisation                                                      #
    # ------------------------------------------------------------------ #

    def _normalise(self, arr: np.ndarray, var: str) -> np.ndarray:
        s   = self._stats[var]
        if s.get("transform") == "quantile" and var in self._quantile_transforms:
            qt    = self._quantile_transforms[var]
            shape = arr.shape
            arr   = qt.transform(arr.ravel().reshape(-1, 1)).ravel().reshape(shape)
        return ((arr - s["mean"]) / (s["std"] + 1e-8)).astype(np.float32)


    def denormalise(self, arr: np.ndarray, var: str) -> np.ndarray:
        s   = self._stats[var]
        out = (arr * (s["std"] + 1e-8) + s["mean"]).astype(np.float32)
        if s.get("transform") == "quantile" and var in self._quantile_transforms:
            qt    = self._quantile_transforms[var]
            shape = out.shape
            out   = qt.inverse_transform(
                out.ravel().reshape(-1, 1)
            ).ravel().reshape(shape).astype(np.float32)
        return out

    # Backward-compatible aliases
    def denormalise_helio(self, arr: np.ndarray, var: str) -> np.ndarray:
        return self.denormalise(arr, var)

    def denormalise_barra(self, arr: np.ndarray, var: str) -> np.ndarray:
        return self.denormalise(arr, var)

    # ------------------------------------------------------------------ #
    # Dataset protocol                                                     #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.start_times)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()

        start = self.start_times[idx]

        i0 = int(np.searchsorted(self._time_array, start))
        i1 = i0 + self.total_length

        if i0 >= len(self._time_array) or self._time_array[i0] != start:
            raise KeyError(
                f"Timestamp {start} not found in {self.zarr_path.name}. "
                f"Check that the index parquet matches the Zarr store."
            )

        T = self.total_length
        H, W = self._ds[self.all_vars[0]].shape[1], self._ds[self.all_vars[0]].shape[2]
        data = np.empty((T, len(self.all_vars), H, W), dtype=np.float32)
        
        for c, var in enumerate(self.all_vars):
            data[:, c] = self._normalise(self._ds[var][i0:i1].astype(np.float32), var)

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
        index_path               = data_cfg["index"][split],
        zarr_path                = data_cfg["zarr"][split],
        stats_path               = data_cfg["stats"]["combined"],
        heliosat_vars            = data_cfg["heliosat_vars"],
        barra_vars               = data_cfg["barra_vars"],
        context_length           = data_cfg["context_length"],
        forecast_length          = data_cfg["forecast_length"],
        satellite_timestep       = f"{data_cfg['satellite_timestep_min']}min",
        quantile_transform_path  = data_cfg["stats"].get("quantile_transforms"),
    )

    sampler = None
    if dist.is_initialized():
        sampler  = DistributedSampler(dataset, shuffle=(split == "train"))
        shuffle  = False  # shuffle is handled by the sampler

    return DataLoader(
        dataset,
        batch_size         = config["training"]["batch_size"][f"stage{config['training']['stage']}"],
        shuffle            = shuffle,
        sampler            = sampler,
        num_workers        = data_cfg["num_workers"],
        pin_memory         = data_cfg.get("pin_memory", True) and data_cfg["num_workers"] > 0,
        prefetch_factor    = data_cfg.get("prefetch_factor", 2) if data_cfg["num_workers"] > 0 else None,
        worker_init_fn     = _worker_init_fn,
        persistent_workers = data_cfg["num_workers"] > 0,
        drop_last          = True,  # drop last batch to avoid size mismatch in distributed training
    )
