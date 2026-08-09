"""
Data loading utilities for satellite solar nowcasting.

Design
------
CPDiTDataset        — main dataset class. Reads the precomputed valid-timestamp
                      index (parquet), opens one or more Zarr stores per split
                      (e.g. one per month), and routes each timestamp lookup to
                      the correct store. This removes the need for a slow
                      combine_zarr step — monthly stores are read directly.

                      Each store is opened once per worker on the first
                      __getitem__ call and cached for the lifetime of that
                      worker, giving O(1) per-sample time-slice access.

build_dataloader    — convenience factory that constructs a CPDiTDataset and
                      wraps it in a DataLoader with the correct settings.
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

    Accepts one or more Zarr stores per split so that monthly files can be
    read directly without a slow combine step. Each timestamp in the
    valid-index is mapped to the store that contains it, and sequences that
    span a store boundary are excluded automatically during index build.

    On __getitem__ the dataset:
      1. Looks up the start timestamp to find which store and integer offset
         to use — O(1), no searching at runtime.
      2. Reads exactly (context_length + forecast_length) timesteps for
         each variable from that store.
      3. Normalises every variable to zero mean / unit variance.
      4. Stacks all variables along the channel axis and returns
         (context, forecast) tensors of shape (T, C, H, W) each.
    """

    def __init__(
        self,
        index_path:                    str | Path,
        zarr_paths:                    List[str | Path],
        stats_path_heliosat:           str | Path,
        stats_path_barra:              str | Path,
        heliosat_vars:                 List[str],
        barra_vars:                    List[str],
        context_length:                int = 12,
        forecast_length:               int = 6,
        satellite_timestep:            str = "10min",
        barra_quantile_transform_path: str | Path | None = None,
    ):
        """
        Args:
            index_path:       Parquet file of valid start timestamps.
            zarr_paths:       List of Zarr store paths (e.g. one per month).
                              Order does not matter — timestamps are matched
                              to stores automatically.
            stats_path:       JSON file of per-variable normalisation stats.
            heliosat_vars:    Heliosat variable names to load.
            barra_vars:       BARRA variable names to load.
            context_length:   Number of context timesteps.
            forecast_length:  Number of forecast timesteps.
            satellite_timestep: Temporal resolution of the data (e.g. "10min").
            barra_quantile_transform_path:
                              Optional path to pickled QuantileTransformer dict.
        """
        self.zarr_paths                    = [Path(p) for p in zarr_paths]
        self.stats_path_heliosat           = Path(stats_path_heliosat)
        self.stats_path_barra              = Path(stats_path_barra)
        self.heliosat_vars                 = heliosat_vars
        self.barra_vars                    = barra_vars
        self.all_vars                      = heliosat_vars + barra_vars
        self.context_length                = context_length
        self.forecast_length               = forecast_length
        self.total_length                  = context_length + forecast_length
        self.satellite_timestep            = pd.Timedelta(satellite_timestep)
        self.barra_quantile_transform_path = Path(barra_quantile_transform_path)

        for p in self.zarr_paths:
            if not p.exists():
                raise FileNotFoundError(
                    f"Zarr store not found: {p}\n"
                    f"Run scripts/prepare_data.py first."
                )

        if not self.stats_path_heliosat.exists():
            raise FileNotFoundError(
                f"Stats file not found: {self.stats_path_heliosat}"
            )

        # Load normalisation stats
        with open(Path(stats_path_heliosat)) as f:
            helio_stats: Dict[str, Dict] = json.load(f)
        with open(Path(stats_path_barra)) as f:
            barra_stats: Dict[str, Dict] = json.load(f)

        # Merge into a single lookup used by _normalise / denormalise.
        # BARRA entries overwrite heliosat entries on key collision (there
        # should be none in practice).
        self._stats: Dict[str, Dict] = {**helio_stats, **barra_stats}

        # Load quantile transforms
        self._quantile_transforms: Dict = {}
        if barra_quantile_transform_path is not None:
            qt_path = Path(barra_quantile_transform_path)
            if not qt_path.exists():
                raise FileNotFoundError(
                    f"Quantile transform file not found: {qt_path}"
                )
            with open(qt_path, "rb") as f:
                self._quantile_transforms = pickle.load(f)

        for var in self.barra_vars:
            s = self._stats.get(var, {})
            if s.get("transform") == "quantile" and var not in self._quantile_transforms:
                warnings.warn(
                    f"Variable '{var}' has transform='quantile' but no quantile "
                    f"transform was loaded. Falling back to z-score only. "
                    f"Pass barra_quantile_transform_path to fix this.",
                    UserWarning,
                    stacklevel=2,
                )

        # Build the timestamp → (store_index, integer_offset) map.
        # Done at construction time (main process) so workers inherit it
        # without repeating the work.
        #
        # We also filter the valid-index to only keep timestamps where the
        # full sequence [start, start + total_length) fits inside a single
        # store, so __getitem__ never has to stitch across a boundary.
        self._ts_to_store_and_idx: Dict[pd.Timestamp, Tuple[int, int]] = {}
        self._build_time_index()

        index = pd.read_parquet(index_path)
        all_starts = pd.DatetimeIndex(index["start_time"].values)

        valid_mask  = [t in self._ts_to_store_and_idx for t in all_starts]
        n_dropped   = (~np.array(valid_mask)).sum()
        if n_dropped > 0:
            warnings.warn(
                f"{n_dropped} timestamps dropped because their full sequence "
                f"({self.total_length} steps) crosses a Zarr store boundary "
                f"or is not present in any store.",
                UserWarning,
                stacklevel=2,
            )

        self.start_times = all_starts[valid_mask]

        # Zarr stores are opened lazily inside workers
        self._stores: List[zarr.Group | None] = [None] * len(self.zarr_paths)

    # ------------------------------------------------------------------ #
    # Time index construction (main process, called once at __init__)     #
    # ------------------------------------------------------------------ #
    def _build_time_index(self) -> None:
        """
        Build a global timestamp → (store_idx, integer_offset) map across
        all stores. A timestamp is registered if its full window of
        total_length steps is available, even if it spans two adjacent stores.
        """
        DATA_EPOCH = pd.Timestamp("2000-01-01")
    
        # Read every store's time array once and keep them in order
        store_times: List[pd.DatetimeIndex] = []
        for path in self.zarr_paths:
            z     = zarr.open(str(path), mode="r")
            times = DATA_EPOCH + pd.to_timedelta(
                z["time"][:].astype(np.int64), unit="s"
            )
            store_times.append(times)
    
        # Build a flat global index: timestamp → (store_idx, offset_in_store)
        # This is used for the first timestep of each window only.
        global_ts_to_loc: Dict[pd.Timestamp, Tuple[int, int]] = {}
        for store_idx, times in enumerate(store_times):
            for i, t in enumerate(times):
                if t not in global_ts_to_loc:   # first store wins on overlap
                    global_ts_to_loc[t] = (store_idx, i)
    
        # Build a flat global timeline for contiguity checks
        # Maps timestamp → global integer position
        all_times_sorted = sorted(global_ts_to_loc.keys())
        global_pos       = {t: i for i, t in enumerate(all_times_sorted)}
    
        # Register a start timestamp only if all total_length steps exist
        # and are contiguous (no gaps) in the global timeline
        for t_start, (store_idx, offset) in global_ts_to_loc.items():
            g0 = global_pos[t_start]
            g1 = g0 + self.total_length
    
            # Check all required timestamps exist
            required = all_times_sorted[g0:g1]
            if len(required) < self.total_length:
                continue
    
            # Check they are evenly spaced (no gaps)
            diffs = pd.DatetimeIndex(required).to_series().diff().dropna()
            if not (diffs == self.satellite_timestep).all():
                continue
    
            self._ts_to_store_and_idx[t_start] = (store_idx, offset)

    # ------------------------------------------------------------------ #
    # Lazy store opening (once per worker)                                #
    # ------------------------------------------------------------------ #

    def _ensure_open(self, store_idx: int) -> zarr.Group:
        if self._stores[store_idx] is None:
            self._stores[store_idx] = zarr.open(
                str(self.zarr_paths[store_idx]), mode="r"
            )
        return self._stores[store_idx]

    # ------------------------------------------------------------------ #
    # Normalisation                                                        #
    # ------------------------------------------------------------------ #

    def _normalise(self, arr: np.ndarray, var: str) -> np.ndarray:
        s   = self._stats[var]
        out = arr.copy()
        if s.get("transform") == "quantile" and var in self._quantile_transforms:
            qt    = self._quantile_transforms[var]
            shape = out.shape
            out   = qt.transform(out.ravel().reshape(-1, 1)).ravel().reshape(shape)
        return ((out - s["mean"]) / (s["std"] + 1e-8)).astype(np.float32)

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
        start = self.start_times[idx]
        store_idx, i0 = self._ts_to_store_and_idx[start]
        i1 = i0 + self.total_length
    
        ds        = self._ensure_open(store_idx)
        store_len = ds["time"].shape[0]
    
        if i1 <= store_len:
            # Common case: entire window fits in one store
            channel_arrays = [
                self._normalise(ds[var][i0:i1].astype(np.float32), var)
                for var in self.all_vars
            ]
        else:
            # Window spans two stores — read tail of current, head of next
            n_this = store_len - i0
            n_next = self.total_length - n_this
    
            if store_idx + 1 >= len(self.zarr_paths):
                raise RuntimeError(
                    f"Window for {start} runs past the last store — "
                    f"this timestamp should have been filtered during index build."
                )
    
            ds_next = self._ensure_open(store_idx + 1)
    
            channel_arrays = [
                self._normalise(
                    np.concatenate([
                        ds[var][i0:].astype(np.float32),
                        ds_next[var][:n_next].astype(np.float32),
                    ], axis=0),
                    var,
                )
                for var in self.all_vars
            ]
    
        data = np.stack(channel_arrays, axis=1)   # (T, C, H, W)
    
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
            f"n_stores={len(self.zarr_paths)}, "
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
        zarr_paths                    = data_cfg["zarr_paths"][split],
        stats_path_heliosat           = data_cfg["normalisation_stats"]["heliosat"],
        stats_path_barra              = data_cfg["normalisation_stats"]["barra"],
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
