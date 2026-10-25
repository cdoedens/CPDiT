"""Data loading via the pyearthtools pipeline for the CPDiT training script."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import math
import os
import random
import traceback
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import xarray as xr
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

import pyearthtools.data as petdata
import pyearthtools.pipeline as petpipe

# Defined with the config so the channel count can be derived without importing
# pyearthtools; re-exported here, where the data pipeline's users look for it.
from src.training.config import channel_names

my_site = 'site_archive_nci'  # 'site_archive_nci' | 'site_archive_jasmin' | 'site_archive_met_office'
import importlib

_ = importlib.import_module(my_site)

import warnings

warnings.filterwarnings("ignore", message="In a future version of xarray the default value for join", category=FutureWarning)
warnings.filterwarnings("ignore", message="In a future version of xarray the default value for compat", category=FutureWarning)
warnings.filterwarnings("ignore", message="Data requested at a higher resolution than available")
warnings.filterwarnings("ignore", message="Importing `spectral_angle_mapper` from `torchmetrics.functional`", category=FutureWarning)

logger = logging.getLogger(__name__)

SAT_TIMESTEP_MINUTES = 10

# Yielded instead of real tensors while priming. Collating these is free, which
# keeps the priming loop measuring archive throughput rather than memcpy.
_PRIME_PLACEHOLDER = torch.zeros(1)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def _parse_timestep(timestep) -> tuple[int, str]:
    """Parse a config timestep ("10 minutes" or (10, "minutes")) into (value, unit)."""
    if isinstance(timestep, (tuple, list)):
        return int(timestep[0]), str(timestep[1])
    parts = str(timestep).split()
    if len(parts) != 2:
        raise ValueError(
            f"Cannot parse timestep {timestep!r}; expected '<value> <unit>', e.g. '10 minutes'."
        )
    return int(parts[0]), parts[1]


def blocked_utc_hours(data_cfg: dict) -> set[int]:
    """
    UTC hours to skip entirely, from ``data.skip_utc_hours: [lo, hi]`` inclusive.

    A sample needs its whole context window in daylight, and outside that window
    the irradiance field is absent or NaN, so those dates cost a full archive
    fetch and yield nothing. Measured over this domain (145-151.5E, UTC+10):
    UTC hours 09-21 produced 0 usable samples out of 26 attempts, while 22-08
    (local 08:00-18:00) produced all of them.

    The range wraps past midnight when lo > hi. Empty/None keeps every hour.

    Note this is a fixed clock window, not a solar calculation: it is chosen to
    sit inside daylight year-round, so in summer it discards some genuinely
    usable early-morning and late-evening samples. Widen it if you want those.
    """
    spec = data_cfg.get("skip_utc_hours")
    if not spec:
        return set()
    lo, hi = int(spec[0]), int(spec[1])
    for h in (lo, hi):
        if not 0 <= h <= 23:
            raise ValueError(f"data.skip_utc_hours must be hours in 0-23, got {spec}.")
    if lo <= hi:
        return set(range(lo, hi + 1))
    return set(range(lo, 24)) | set(range(0, hi + 1))    # wraps midnight


def gap_utc_times(data_cfg: dict) -> set[str]:
    """
    UTC clock times (``"HH:MM"``) from ``data.interpolate_gap_times`` whose
    frame, when missing, is filled by linear interpolation between the frames
    one step either side.

    Himawari skips the 02:40 UTC scan every day for housekeeping. Without
    filling it, every sample whose window touches 02:40 is lost -- 18
    consecutive anchors, three hours of local midday, every day. Empty/None
    disables interpolation.
    """
    times = set()
    for spec in data_cfg.get("interpolate_gap_times") or []:
        try:
            times.add(pd.Timestamp(f"2000-01-01 {spec}").strftime("%H:%M"))
        except ValueError as exc:
            raise ValueError(
                f"data.interpolate_gap_times entries must be 'HH:MM' UTC, got {spec!r}."
            ) from exc
    return times


def frame_times(anchor: str, context_length: int, forecast_length: int,
                step_minutes: int = SAT_TIMESTEP_MINUTES) -> list[str]:
    """
    Timestamps of every frame in one sample, as PET-format strings.

    ``context_length`` frames immediately *before* the anchor, then
    ``forecast_length`` frames starting *at* it — the same window the
    TemporalWindow-plus-trim used to produce, stated directly.
    """
    t0 = pd.Timestamp(anchor)
    offsets = list(range(-context_length, forecast_length))
    return [
        (t0 + pd.Timedelta(minutes=step_minutes * k)).strftime("%Y%m%dT%H%M")
        for k in offsets
    ]


def nearest_hour(stamp: str) -> str:
    """
    Round a timestamp to the nearest hour.

    BARRA is hourly and the archive floors a request to the containing hour, so
    rounding here reproduces the ``method='nearest'`` temporal interpolation the
    windowed pipeline used to do — and makes each BARRA hour reusable across the
    six satellite frames that share it.

    Ties round *down*, matching scipy's 'nearest' (which xarray uses) rather
    than pandas' ``.round``, which is banker's rounding and would send 12:30 to
    12:00 but 13:30 to 14:00. Satellite frames land exactly on :30 once an hour,
    so this tie-break decides one frame in six.
    """
    t = pd.Timestamp(stamp) + pd.Timedelta(minutes=30) - pd.Timedelta(nanoseconds=1)
    return t.floor("h").strftime("%Y%m%dT%H%M")


def _build_date_range(
    split:       str,
    config:      dict,
    shard_index: int = 0,
    num_shards:  int = 1,
):
    """
    The anchor dates this shard is responsible for.

    Sharding is applied here rather than by filtering a stream of fetched
    samples: each shard starts `shard_index` steps into the range and strides by
    `num_shards`, so the shards form a disjoint cover while each one only ever
    requests what it will yield. Filtering fetched output instead would still
    pull every discarded sample through the archive.

    `sample_stride` folds into the same arithmetic. Returns None when the
    shard's offset start falls past the end of the split (the tail shards of a
    short range), which `DateRange` would otherwise reject as an inverted range.
    """
    data_cfg  = config["data"]
    split_cfg = data_cfg["splits"][split]
    start_date, end_date = split_cfg["start"], split_cfg["end"]

    step_value, step_unit = _parse_timestep(data_cfg.get("sat_timestep", "10 minutes"))
    sample_stride = max(1, int(data_cfg.get("sample_stride", 1)))
    shard_step    = step_value * sample_stride

    iterator_start = start_date
    if shard_index:
        iterator_start = str(
            petdata.time.Petdt(start_date)
            + petdata.time.TimeDelta((shard_step * shard_index, step_unit))
        )
        if petdata.time.Petdt(iterator_start) > petdata.time.Petdt(end_date):
            return None

    return petpipe.iterators.DateRange(
        iterator_start, end_date, interval=(shard_step * num_shards, step_unit)
    )


def _build_frame_pipelines(config: dict) -> tuple[petpipe.Pipeline, petpipe.Pipeline]:
    """
    Two pipelines that each return a SINGLE timestamp.

    This is the change that makes frame-level caching possible. The previous
    design fetched a whole 15-frame window per sample via TemporalWindow, but
    consecutive samples overlap heavily — at sample_stride 3 the window advances
    3 frames and so each frame was decoded about 5 times per epoch. Fetching
    frames individually lets the cache store each timestamp once, which removes
    that redundancy from the expensive cold pass.

    Must be called inside the consuming process: the pipeline is not fork-safe.
    """
    data_cfg = config["data"]
    bounds   = data_cfg.get("bounds", [-35, -28.5, 145, 151.5])

    sat = petpipe.Pipeline(
        petdata.archive.Himawari(
            data_cfg.get("himawari_vars", ["surface_global_irradiance"])
        ),
        petpipe.operations.xarray.Sort(order=['time', 'latitude', 'longitude']),
        # Align the data variable's coordinate order to the dataset coordinate
        # order so all arrays come back with the same shape.
        petpipe.operations.xarray.AlignDataVariableDimensionsToDatasetCoords(),
        petdata.transform.region.Bounding(*bounds),
        exceptions_to_ignore=petdata.exceptions.DataNotFoundError,
    )

    bar = petpipe.Pipeline(
        petdata.archive.BARRA_V2(
            data_cfg.get("barra_vars", ["RH24mean"]),
            domain_id = data_cfg.get("barra_domain", "AUST-11"),
            frequency = data_cfg.get("barra_freq", "1hr"),
        ),
        petdata.transforms.coordinates.Drop("crs"),
        petpipe.operations.xarray.Sort(order=['time', 'latitude', 'longitude']),
        petpipe.operations.xarray.AlignDataVariableDimensionsToDatasetCoords(),
        petdata.transform.region.Bounding(*bounds),
        exceptions_to_ignore=petdata.exceptions.DataNotFoundError,
    )
    return sat, bar


# ---------------------------------------------------------------------------
# Raw Himawari bands
# ---------------------------------------------------------------------------
# The AHI band files are not on a lat/lon grid. They sit on the satellite's
# fixed geostationary grid (5500 x 5500 at 2 km, 22000 x 22000 for B03 at
# 500 m), so `Bounding` cannot cut them and they cannot be `interp`'d onto the
# irradiance grid by coordinate.
#
# The ancillary GEOM_SENSOR files store a lat/lon for every native pixel, and
# a KD-tree over them recovers the nearest native pixel for each model pixel.
# That works but means scanning ~4 GB of lat/lon for the 500 m grid. Instead
# the model grid is projected forward into the satellite's own coordinates
# with the projection stored in each band file, which makes the nearest native
# pixel a rounding. MEASURED over the training domain: identical to the
# ancillary KD-tree for 97% of pixels and never more than one pixel off (the
# KD-tree measures distance in degrees, not on the projected grid, so it is
# the less accurate of the two at ties).
#
# Every band is reduced onto the 2 km grid before sampling. The 1 km and
# 500 m pixels nest exactly inside the 2 km ones, so a block mean is a proper
# area average, and the model grid (0.02 deg, ~2 km) is too coarse to resolve
# anything finer -- nearest-neighbour sampling at 500 m would only alias.

BAND_GRID_METRES = 2000
# Native resolution of the bands that are not at 2 km.
_BAND_RESOLUTION = {"B01": 1000, "B02": 1000, "B03": 500}


def band_groups(bands: list[str]) -> list[list[str]]:
    """
    Split bands into the groups one `HimawariChannels` accessor can serve.

    The accessor refuses to mix resolutions. Products are separated too: the
    file variables are named by channel number alone, so ``OBS_B03`` and
    ``BRF_B03`` would collide in one dataset.
    """
    groups: dict[tuple[str, int], list[str]] = {}
    for band in bands:
        product, number = band.rsplit("_", 1)
        key = (product, _BAND_RESOLUTION.get(number, BAND_GRID_METRES))
        groups.setdefault(key, []).append(band)
    return list(groups.values())


def _build_band_accessors(config: dict) -> list[tuple[list[str], petdata.archive.HimawariChannels]]:
    """One (bands, accessor) pair per resolution/product group."""
    bands = list(config["data"].get("himawari_bands", []))
    return [(group, petdata.archive.HimawariChannels(bands=group))
            for group in band_groups(bands)]


def _band_variable(ds: xr.Dataset, band: str) -> str:
    """The file variable holding a band, e.g. OBS_B13 -> channel_0013_brightness_temperature."""
    prefix = f"channel_{int(band.rsplit('_B', 1)[1]):04d}_"
    matches = [v for v in ds.data_vars if str(v).startswith(prefix)]
    if len(matches) != 1:
        # Deterministic for this timestamp, so reported as missing data and the
        # frame is remembered as bad rather than retried every epoch.
        raise petdata.exceptions.DataNotFoundError(
            f"Expected one variable for band {band} ({prefix}*), found {matches}."
        )
    return matches[0]


class BandRegridder:
    """
    Nearest native pixel for every model pixel, for any AHI resolution.

    The index arrays depend only on the model grid and the native grid, so
    they are built once per resolution and reused for every frame.
    """

    def __init__(self, latitude: np.ndarray, longitude: np.ndarray):
        self.latitude  = np.asarray(latitude)
        self.longitude = np.asarray(longitude)
        self._maps: dict = {}

    def matches(self, latitude: np.ndarray, longitude: np.ndarray) -> bool:
        return (np.array_equal(self.latitude, latitude)
                and np.array_equal(self.longitude, longitude))

    def _index(self, x: np.ndarray, y: np.ndarray, proj4: str):
        dx, dy = float(x[1] - x[0]), float(y[1] - y[0])       # y descends: dy < 0
        key = (proj4, float(x[0]), float(y[0]), dx, dy)
        if key in self._maps:
            return self._maps[key]

        import pyproj

        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", pyproj.CRS.from_proj4(proj4), always_xy=True
        )
        lat, lon = np.meshgrid(self.latitude, self.longitude, indexing="ij")
        px, py = transformer.transform(lon, lat)
        if not (np.isfinite(px).all() and np.isfinite(py).all()):
            raise ValueError("Part of the model grid lies off the Himawari disk.")

        # Index of the 2 km cell containing each model pixel, counted from the
        # outer edge of the native grid, which every resolution shares.
        factor = int(round(BAND_GRID_METRES / abs(dx)))
        yb = np.floor((py - (y[0] - dy / 2)) / (dy * factor)).astype(np.int64)
        xb = np.floor((px - (x[0] - dx / 2)) / (dx * factor)).astype(np.int64)
        y0, x0 = int(yb.min()), int(xb.min())
        entry = (
            slice(y0 * factor, (int(yb.max()) + 1) * factor),
            slice(x0 * factor, (int(xb.max()) + 1) * factor),
            factor, yb - y0, xb - x0,
        )
        self._maps[key] = entry
        return entry

    def __call__(self, da: xr.DataArray, proj4: str) -> np.ndarray:
        """Resample one native (y, x) band onto the model grid."""
        y_slice, x_slice, factor, yi, xi = self._index(da["x"].values, da["y"].values, proj4)
        # Only the box under the model grid is read from disk.
        arr = da.isel(y=y_slice, x=x_slice).values.astype(np.float32)
        if factor > 1:
            h, w = arr.shape
            arr = arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
        return arr[yi, xi]


def _read_bands(accessor, stamp: str) -> xr.Dataset:
    """
    Open one timestamp of a band group, lazily and in the file's own chunks.

    By default pyearthtools opens with ``chunks='auto'``, which on the 500 m
    file gives full-width (1520, 22000) dask chunks: reading the ~1000 x 1000
    box under the model grid then decodes most of the 1.9 GB disk, MEASURED
    11 s per frame. Without dask-level chunks each read only touches the
    file's own (20, 2000) chunks it overlaps, ~1 s.
    """
    import pyearthtools.utils

    try:
        with pyearthtools.utils.config.set({"data.open.xarray.chunks": None,
                                            "data.open.xarray_mf.chunks": None}):
            return accessor[stamp]
    except FileNotFoundError as exc:
        # The accessor lists the timestamp's directory directly, so a missing
        # scan surfaces as FileNotFoundError rather than DataNotFoundError.
        raise petdata.exceptions.DataNotFoundError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Transparent frame cache
# ---------------------------------------------------------------------------
# Building one sample from the archive costs ~16-20s, which is ~7x slower than
# the GPU consumes samples, so an uncached run leaves the GPU ~93% idle. The
# work is entirely deterministic — the same timestamp and the same config always
# give the same array — so it is perfectly cacheable.
#
# The unit of caching is a single FRAME, not an assembled sample. Samples
# overlap heavily: the window is context_length + forecast_length frames but
# advances only sample_stride frames, so at the defaults each frame belongs to
# about five different samples. Caching assembled samples would store every
# frame five times and, worse, decode it five times on the cold pass. Caching
# frames stores and decodes each exactly once: ~11 GB per year instead of
# ~56 GB, and a correspondingly cheaper first pass.
#
# The cache key hashes every config value that changes the tensor, including
# the normalisation statistics (which are baked into the stored array). Change
# a variable, the bounds, the window, the crop or the stats and you get a new
# namespace automatically — so the cache can never silently serve data built
# from a different recipe. That keeps the "edit the config freely" workflow
# intact without a manual invalidation step.


def _cache_namespace(config: dict) -> str:
    """Hash of everything that affects the produced tensor."""
    data_cfg = config["data"]
    stats_path = data_cfg["stats_path"]
    with open(stats_path) as f:
        stats = json.load(f)
    # Only the statistics of channels this config uses. Adding statistics for
    # a new variable to the shared file must not invalidate caches built
    # without it.
    names = set(channel_names(config))
    stats = {k: v for k, v in stats.items() if k in names}

    recipe = {
        "himawari_vars": list(data_cfg.get("himawari_vars", [])),
        "barra_vars":    list(data_cfg.get("barra_vars", [])),
        "barra_domain":  data_cfg.get("barra_domain"),
        "barra_freq":    data_cfg.get("barra_freq"),
        "bounds":        list(data_cfg.get("bounds", [])),
        "crop_offset":   data_cfg.get("crop_offset", 10),
        "image_size":    config["model"].get("image_size", 256),
        # NB: n_prior_sat / n_post are deliberately NOT part of the key. A
        # frame is independent of the window it lands in, so changing the
        # context or forecast length reuses the existing frames instead of
        # rebuilding the cache from scratch.
        "sat_timestep":  data_cfg.get("sat_timestep", "10 minutes"),
        # Normalisation is applied before caching, so the statistics are part
        # of the recipe. Recomputing stats must invalidate the cache.
        "stats":         {k: [v["mean"], v["std"]] for k, v in sorted(stats.items())},
        # 3: channels stacked in the configured order. Version 2 frames were
        #    stacked in the archive's `data_vars` order, which varies between
        #    fetches, so a fraction of them have their channels permuted.
        "version":       3,
    }
    # Added only when set, so configs without bands keep their existing cache.
    bands = list(data_cfg.get("himawari_bands", []))
    if bands:
        recipe["himawari_bands"] = bands
    blob = json.dumps(recipe, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def resolve_cache_dir(config: dict) -> Optional[Path]:
    """
    Resolve dataloader.cache_dir, expanding environment variables.

    ``$PBS_JOBFS`` is local NVMe and the fastest option, but it is destroyed
    when the job ends, so every job re-pays the first epoch. A /scratch path
    persists across jobs and is still ~1000x faster than the archive, which is
    the better default: the archive cost is then paid once, ever.
    """
    loader_cfg = config.get("dataloader", {})
    raw = loader_cfg.get("cache_dir")
    if not raw:
        return None
    expanded = os.path.expandvars(str(raw))
    if "$" in expanded:
        logger.warning(
            "cache_dir %r contains an unset environment variable (expanded to "
            "%r); caching disabled.", raw, expanded
        )
        return None
    return Path(expanded)


class FrameCache:
    """
    Content-addressed cache of processed frames, one .npy per timestamp.

    One file per sample rather than one big store: workers write concurrently
    with no coordination, writes are made atomic with a rename, and a partially
    written cache is simply a partially warm one.
    """

    def __init__(self, root: Optional[Path], split: str, dtype: str = "float16"):
        self.root = (root / split) if root is not None else None
        self.dtype = np.dtype(dtype)
        self.hits = 0
        self.misses = 0
        self.known_bad = 0
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def _paths(self, date) -> tuple[Path, Path]:
        # Day-level subdirectories: a year at a 10-minute cadence is ~50k files,
        # which is slow to stat in a single directory.
        stamp = str(date).replace("-", "").replace(":", "").replace("T", "")[:12]
        day = self.root / stamp[:8]
        return day / f"{stamp}.npy", day / f"{stamp}.bad"

    def _interp_path(self, date) -> Path:
        # Interpolated frames live beside the frame's `.bad` marker under their
        # own name, so they are served only while gap interpolation is enabled
        # and switching it off needs no new cache namespace.
        path, _ = self._paths(date)
        return path.with_name(path.stem + ".interp.npy")

    def has_interpolated(self, date) -> bool:
        return self.enabled and self._interp_path(date).exists()

    def load_interpolated(self, date) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        path = self._interp_path(date)
        try:
            arr = np.load(path)
        except (FileNotFoundError, NotADirectoryError):
            return None
        except Exception as exc:  # noqa: BLE001 - a corrupt entry must not be fatal
            logger.warning("Discarding unreadable cache entry %s: %s", path, exc)
            path.unlink(missing_ok=True)
            return None
        return torch.from_numpy(arr).float()

    def store_interpolated(self, date, tensor: torch.Tensor) -> None:
        if not self.enabled:
            return
        self._atomic_write(self._interp_path(date), tensor.numpy().astype(self.dtype))

    def exists(self, date) -> tuple[bool, bool]:
        """
        Return (frame_present, known_bad) using stat() only.

        Priming only needs each frame to BE on disk; it never looks at the
        values. Reading a cached frame back costs ~3 ms, which is nothing per
        frame but ~26 minutes of pure re-reading across a 60k-frame cache — paid
        again on every re-run, and priming gets re-run every time a job hits
        walltime. A stat() is ~1000x cheaper and answers the only question that
        matters here.
        """
        if not self.enabled:
            return False, False
        path, bad = self._paths(date)
        if bad.exists():
            self.known_bad += 1
            return False, True
        if path.exists():
            self.hits += 1
            return True, False
        return False, False

    def load(self, date) -> tuple[Optional[torch.Tensor], bool]:
        """Return (tensor, known_bad). Both None/False on a miss."""
        if not self.enabled:
            return None, False
        path, bad = self._paths(date)
        if bad.exists():
            self.known_bad += 1
            return None, True
        try:
            arr = np.load(path)
        except (FileNotFoundError, NotADirectoryError):
            self.misses += 1
            return None, False
        except Exception as exc:  # noqa: BLE001 - a corrupt entry must not be fatal
            logger.warning("Discarding unreadable cache entry %s: %s", path, exc)
            path.unlink(missing_ok=True)
            self.misses += 1
            return None, False
        self.hits += 1
        return torch.from_numpy(arr).float(), False

    def store(self, date, tensor: torch.Tensor) -> None:
        """Store one (C, H, W) frame."""
        if not self.enabled:
            return
        path, _ = self._paths(date)
        self._atomic_write(path, tensor.numpy().astype(self.dtype))

    def store_bad(self, date) -> None:
        """
        Remember that a timestamp yields nothing (missing file, or all-NaN at
        night), so later epochs do not re-pay the fetch to discard it again.
        Marking a single bad frame also rejects every sample containing it,
        instantly.
        """
        if not self.enabled:
            return
        _, bad = self._paths(date)
        self._atomic_write(bad, np.zeros(0, dtype=np.uint8))

    @staticmethod
    def _atomic_write(path: Path, arr: np.ndarray) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
            np.save(tmp, arr, allow_pickle=False)
            # np.save appends .npy if the suffix is missing.
            written = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npy")
            os.replace(written, path)
        except OSError as exc:
            # A full or unwritable cache must degrade to "no cache", never kill
            # the run.
            logger.warning("Cache write failed for %s: %s", path, exc)

    def summary(self) -> str:
        total = self.hits + self.misses + self.known_bad
        rate = 100 * self.hits / total if total else 0.0
        return (f"cache hits={self.hits} misses={self.misses} "
                f"known-bad={self.known_bad} ({rate:.0f}% hit)")


# ---------------------------------------------------------------------------
# Sharding helpers
# ---------------------------------------------------------------------------

def _shard_info() -> tuple[int, int]:
    """
    Return (shard_index, num_shards) covering both DDP ranks and DataLoader
    workers.

    An IterableDataset yields the same stream in every process unless it is
    explicitly sharded, so without this every DDP rank would train on identical
    batches and the extra GPUs would buy nothing.
    """
    rank, world_size = 0, 1
    if dist.is_available() and dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()

    worker_id, num_workers = 0, 1
    info = get_worker_info()
    if info is not None:
        worker_id, num_workers = info.id, info.num_workers

    return rank * num_workers + worker_id, world_size * num_workers


# ---------------------------------------------------------------------------
# IterableDataset wrapper
# ---------------------------------------------------------------------------

class PipelineDataset(IterableDataset):
    """
    Wraps a pyearthtools Pipeline as a PyTorch IterableDataset.

    Each pipeline sample is trimmed to `context_length + forecast_length`
    frames of shape (t, c, h, w) and split into:

        context:  (context_length,  c, h, w) float32
        forecast: (forecast_length, c, h, w) float32

    The split index is explicit rather than "everything but the last frame",
    so extending the forecast horizon cannot silently leak target frames into
    the context.

    Channel order is fixed by `channel_names()`, not by the order the archive
    returns its variables in, so satellite variables come first and
    `irradiance_channel` (default 0) is `himawari_vars[0]`.
    """

    def __init__(self, split: str, config: dict, shuffle: bool = False,
                 prime_mode: bool = False):
        super().__init__()
        self.split  = split
        self.config = config
        # Priming populates the disk cache and never reads the tensors, so it
        # skips the decode/stack entirely and yields a placeholder.
        self.prime_mode = prime_mode

        data_cfg = config["data"]
        self.context_length  = data_cfg.get("n_prior_sat", 12)
        self.forecast_length = data_cfg.get("n_post", 1)
        self.image_size      = config["model"].get("image_size", 256)
        self.crop_offset     = data_cfg.get("crop_offset", 10)
        self.bands           = list(data_cfg.get("himawari_bands", []))

        # Adding a variable or band to the data block without updating the
        # model would otherwise only fail at the VAE's first conv, after the
        # workers have spawned and the first frames have been fetched.
        n_channels = config["model"].get("image_channels")
        if n_channels is not None and int(n_channels) != len(channel_names(config)):
            raise ValueError(
                f"model.image_channels is {n_channels} but the data block produces "
                f"{len(channel_names(config))} channels: {channel_names(config)}."
            )

        # Load normalisation statistics once, not once per epoch.
        with open(data_cfg["stats_path"]) as f:
            self.stats = json.load(f)

        loader_cfg          = config.get("dataloader", {})
        self.shuffle        = shuffle
        self.shuffle_buffer = int(loader_cfg.get("shuffle_buffer", 256)) if shuffle else 0
        # Samples held back before the first yield. Keep this at roughly one
        # batch: it bounds time-to-first-batch, which is pure GPU idle time.
        self.shuffle_min_fill = max(
            1,
            min(
                int(loader_cfg.get("shuffle_min_fill", loader_cfg.get("batch_size", 8))),
                self.shuffle_buffer or 1,
            ),
        )

        self.epoch = 0
        self.last_anchor: Optional[str] = None   # anchor of the most recent yielded sample

        # Transparent cache. The namespace hashes the recipe, so a config change
        # lands in a fresh directory rather than serving stale tensors.
        root = resolve_cache_dir(config)
        self.cache = FrameCache(
            (root / _cache_namespace(config)) if root else None,
            split,
            dtype=loader_cfg.get("cache_dtype", "float16"),
        )

        # UTC hours whose samples are never usable (local night for this domain).
        self.blocked_hours = blocked_utc_hours(data_cfg)

        # UTC clock times of known single-scan gaps, filled by interpolation.
        self.gap_times = gap_utc_times(data_cfg)
        self.n_interpolated = 0
        self.n_skipped_hour = 0

        # Diagnostics — reported once per epoch instead of printed per sample.
        self.n_yielded       = 0
        self.n_skipped_nan   = 0
        self.n_skipped_error = 0
        self.n_skipped_shape = 0

        # A whole era of missing data looks exactly like "slow" from outside:
        # the loader keeps working, yields nothing, and the progress bar sits at
        # zero batches. Counting consecutive failures turns that into a message.
        self.n_consecutive_fail = 0
        self.first_failed_anchor = None
        self._fail_warn_at = self.CONSECUTIVE_FAIL_WARN

    # Consecutive unusable anchors before the stream complains, and the factor
    # by which the threshold grows after each warning so a genuinely sparse
    # period does not spam the log.
    CONSECUTIVE_FAIL_WARN = 200
    CONSECUTIVE_FAIL_BACKOFF = 4

    def set_epoch(self, epoch: int) -> None:
        """Reseed the shuffle buffer so each epoch sees a different order."""
        self.epoch = epoch

    # ------------------------------------------------------------------ #

    def _normalise(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Z-score each variable using the saved statistics.

        The statistics are in raw physical units, so the pipeline must not
        apply any prior scaling to these fields.
        """
        for var in ds.data_vars:
            if var not in self.stats:
                raise KeyError(
                    f"No normalisation statistics for variable '{var}' in "
                    f"{self.config['data']['stats_path']}. "
                    f"Re-run scripts/calc_norm_stats.py."
                )
            mean = self.stats[var]["mean"]
            std  = self.stats[var]["std"]
            ds[var] = (ds[var] - mean) / (std + 1e-8)
        return ds

    def _open_pipelines(self) -> None:
        """
        Build the archive pipelines. Must run in the consuming process: the
        pipeline is not fork-safe.
        """
        self._sat_pipe, self._bar_pipe = _build_frame_pipelines(self.config)
        self._band_accessors = _build_band_accessors(self.config)
        self._band_regridder: Optional[BandRegridder] = None
        self._bar_memo = {}

    def _fetch_bands(self, stamp: str, latitude: np.ndarray,
                     longitude: np.ndarray) -> xr.Dataset:
        """The configured raw bands at one timestamp, in physical units, on the given grid."""
        if self._band_regridder is None or not self._band_regridder.matches(latitude, longitude):
            self._band_regridder = BandRegridder(latitude, longitude)

        out = {}
        for group, accessor in self._band_accessors:
            ds = _read_bands(accessor, stamp)
            proj4 = ds["geostationary"].attrs["proj4"]
            for band in group:
                da = ds[_band_variable(ds, band)].isel(time=0, drop=True)
                out[band] = (("latitude", "longitude"), self._band_regridder(da, proj4))
        return xr.Dataset(out, coords={"latitude": latitude, "longitude": longitude})

    def _fetch_frame(self, stamp: str) -> Optional[torch.Tensor]:
        """
        Build one (C, H, W) frame: fetch, normalise, regrid BARRA, crop, add
        the raw bands, stack.

        Returns None when the frame is unusable (wrong shape, or NaN — which is
        what the irradiance field is at night). Raises only on genuinely
        unexpected errors; missing data is reported as None by the caller.
        """
        sat = self._sat_pipe[stamp].isel(time=0, drop=True)

        # BARRA is hourly and shared by the six satellite frames in each hour,
        # so it is memoised per hour rather than refetched per frame.
        hour = nearest_hour(stamp)
        bar = self._bar_memo.get(hour)
        if bar is None:
            bar = self._bar_pipe[hour].isel(time=0, drop=True)
            if len(self._bar_memo) >= 8:        # a couple of hours either side
                self._bar_memo.clear()
            self._bar_memo[hour] = bar

        sat = self._normalise(sat.copy())
        bar = self._normalise(bar.copy())

        # Spatial regrid only: the temporal 'nearest' is already handled by
        # rounding the request to the nearest hour.
        bar_interp = bar.interp(
            latitude=sat.latitude, longitude=sat.longitude, method="nearest"
        )
        combined = xr.merge([sat, bar_interp])

        o = self.crop_offset
        combined = combined.isel(
            latitude  = slice(o, o + self.image_size),
            longitude = slice(o, o + self.image_size),
        )

        # The bands are sampled straight onto the cropped grid: they have no
        # lat/lon coordinates of their own to bound or crop by.
        if self.bands:
            bands = self._fetch_bands(
                stamp, combined["latitude"].values, combined["longitude"].values
            )
            combined = combined.assign(self._normalise(bands).data_vars)

        # Explicit channel order. `combined.data_vars` is not reproducible --
        # see channel_names() -- so ordering by it gives frames whose channels
        # are permuted at random relative to each other.
        names = channel_names(self.config)
        missing = [v for v in names if v not in combined.data_vars]
        if missing:
            raise KeyError(
                f"Frame {stamp} is missing requested variable(s) {missing}; "
                f"the pipeline returned {list(combined.data_vars)}."
            )
        arr = np.stack([combined[v].values for v in names], axis=0)
        frame = torch.from_numpy(np.ascontiguousarray(arr)).float()   # (C, H, W)

        if tuple(frame.shape[-2:]) != (self.image_size, self.image_size):
            self.n_skipped_shape += 1
            logger.debug("Frame %s has spatial shape %s", stamp, tuple(frame.shape[-2:]))
            return None
        if torch.isnan(frame).any():
            self.n_skipped_nan += 1
            return None
        return frame

    def _ensure_frame(self, stamp: str) -> bool:
        """
        Make sure one frame is on disk, without decoding it.

        The priming counterpart of `_frame`: same fetch-and-store path on a
        miss, but a cache hit costs a stat() instead of a read + tensor
        construction.
        """
        present, known_bad = self.cache.exists(stamp)
        if present:
            return True
        if not known_bad and self._real_frame(stamp) is not None:
            return True
        if not self._is_gap_time(stamp):
            return False
        return self.cache.has_interpolated(stamp) or self._interpolate_gap(stamp) is not None

    def _frame(self, stamp: str) -> Optional[torch.Tensor]:
        """
        Cache-first access to a single frame. A frame missing at one of
        `gap_times` is filled by interpolation (see `_interpolate_gap`).
        """
        frame = self._real_frame(stamp)
        if frame is None and self._is_gap_time(stamp):
            frame = self.cache.load_interpolated(stamp)
            if frame is None:
                frame = self._interpolate_gap(stamp)
        return frame

    def _is_gap_time(self, stamp: str) -> bool:
        return bool(self.gap_times) and pd.Timestamp(stamp).strftime("%H:%M") in self.gap_times

    def _interpolate_gap(self, stamp: str) -> Optional[torch.Tensor]:
        """
        Linear interpolation in time across one missing frame: the mean of the
        frames one step either side, which must both be real (never themselves
        interpolated). Interpolating the z-scored values equals interpolating
        the physical ones, as the normalisation is affine. The result is cached.
        """
        t     = pd.Timestamp(stamp)
        step  = pd.Timedelta(minutes=SAT_TIMESTEP_MINUTES)
        before = self._real_frame((t - step).strftime("%Y%m%dT%H%M"))
        if before is None:
            return None
        after  = self._real_frame((t + step).strftime("%Y%m%dT%H%M"))
        if after is None:
            return None
        frame = 0.5 * (before + after)
        self.cache.store_interpolated(stamp, frame)
        self.n_interpolated += 1
        return frame

    def _real_frame(self, stamp: str) -> Optional[torch.Tensor]:
        """Cache-first access to a single observed frame; never interpolated."""
        frame, known_bad = self.cache.load(stamp)
        if known_bad:
            return None
        if frame is not None:
            return frame

        try:
            frame = self._fetch_frame(stamp)
        except petdata.exceptions.DataNotFoundError as exc:
            self.n_skipped_error += 1
            self.cache.store_bad(stamp)
            logger.debug("Missing frame %s: %s", stamp, exc)
            return None
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill training
            self.n_skipped_error += 1
            logger.warning("Frame %s failed: %s: %s", stamp, type(exc).__name__, exc)
            logger.debug("%s", traceback.format_exc())
            return None

        if frame is None:
            self.cache.store_bad(stamp)      # deterministic, so remember it
            return None

        self.cache.store(stamp, frame)
        return frame

    def _sample(self, anchor: str) -> Optional[torch.Tensor]:
        """
        Assemble (T, C, H, W) for one anchor date from its individual frames.

        Any unusable frame invalidates the whole sample — and because bad frames
        are marked in the cache, later epochs reject it without any I/O.
        """
        stamps = frame_times(anchor, self.context_length, self.forecast_length)

        if self.prime_mode:
            # Only the side effect matters: every frame ends up on disk, or the
            # sample is unusable. Nothing is decoded or stacked.
            for stamp in stamps:
                if not self._ensure_frame(stamp):
                    return None
            return _PRIME_PLACEHOLDER

        frames = []
        for stamp in stamps:
            frame = self._frame(stamp)
            if frame is None:
                return None
            frames.append(frame)
        return torch.stack(frames)

    def _raw_stream(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        Yield (context, forecast) pairs for this process's shard.

        Driven by the anchor date list, so nothing is fetched that will not be
        used: blocked hours are dropped before any I/O, and a fully cached
        sample never touches the archive at all.
        """
        shard_index, num_shards = _shard_info()

        dates = _build_date_range(self.split, self.config, shard_index, num_shards)
        if dates is None:                 # range shorter than num_shards samples
            return

        # Built here, in the consuming process: the pipeline is not fork-safe.
        self._open_pipelines()

        for date in dates:
            stamp = str(date)
            if int(pd.Timestamp(stamp).hour) in self.blocked_hours:
                # Local night: the whole context window is dark, so this anchor
                # could never produce a usable sample. Skipped before any I/O.
                self.n_skipped_hour += 1
                continue

            tensor = self._sample(stamp)
            if tensor is None:
                if self.first_failed_anchor is None:
                    self.first_failed_anchor = stamp
                self.n_consecutive_fail += 1
                if self.n_consecutive_fail >= self._fail_warn_at:
                    logger.warning(
                        "[%s] %d consecutive anchors yielded no usable sample "
                        "(%s .. %s). Nothing is being produced, so this looks "
                        "like slowness rather than a failure. The usual cause is "
                        "a split that starts before the archive does — check "
                        "data.splits.%s.start against what the archive actually "
                        "serves.",
                        self.split, self.n_consecutive_fail,
                        self.first_failed_anchor, stamp, self.split,
                    )
                    self._fail_warn_at *= self.CONSECUTIVE_FAIL_BACKOFF
                continue

            logger.debug("sample %s", stamp)
            if self.prime_mode:
                self.n_consecutive_fail = 0
                self.first_failed_anchor = None
                self._fail_warn_at = self.CONSECUTIVE_FAIL_WARN
                self.n_yielded += 1
                yield _PRIME_PLACEHOLDER, _PRIME_PLACEHOLDER
                continue

            self.n_consecutive_fail = 0
            self.first_failed_anchor = None
            self._fail_warn_at = self.CONSECUTIVE_FAIL_WARN
            self.n_yielded += 1
            self.last_anchor = stamp
            yield tensor[:self.context_length], tensor[self.context_length:]

    def sample_at(
        self, anchor: str, allow_missing_target: bool = False
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """
        Build the (context, forecast) pair for one specific anchor date.

        Bypasses the split's date range, sharding and stride -- it is for
        inspecting a chosen time, not for training. Returns None when a context
        frame is unusable (missing or NaN, e.g. at night).

        With ``allow_missing_target`` an unusable *forecast* frame no longer
        invalidates the sample: it comes back as an all-NaN frame. A forecast
        needs only its context, so a gap in the truth should not stop one being
        made -- the missing frame just has nothing to be compared against.
        """
        self._open_pipelines()
        if not allow_missing_target:
            tensor = self._sample(anchor)
            if tensor is None:
                return None
            return tensor[:self.context_length], tensor[self.context_length:]

        stamps  = frame_times(anchor, self.context_length, self.forecast_length)
        context = [self._frame(stamp) for stamp in stamps[:self.context_length]]
        if any(frame is None for frame in context):
            return None
        target = [self._frame(stamp) for stamp in stamps[self.context_length:]]
        target = [torch.full_like(context[0], float("nan")) if frame is None else frame
                  for frame in target]
        return torch.stack(context), torch.stack(target)

    def __iter__(self):
        stream = self._raw_stream()

        if not self.shuffle or self.shuffle_buffer <= 1:
            yield from stream
            return

        # Reservoir shuffle. The pipeline is strictly date-ordered and windows
        # at a 10-minute cadence overlap heavily, so consecutive samples are
        # near-duplicates; without this the gradient steps within an epoch are
        # almost perfectly correlated.
        #
        # Crucially the buffer is NOT pre-filled before the first yield. A
        # sample costs ~20s of archive I/O, so filling 256 slots first would
        # fetch 264 samples before the loader emitted one batch — about 80
        # minutes per worker with the GPU idle and nothing on screen, which is
        # indistinguishable from a hang. Instead the first sample is emitted
        # after `shuffle_min_fill` intakes, and thereafter one sample is
        # emitted for every two taken in, so the buffer still grows to
        # `shuffle_buffer` but the loader never blocks waiting for it.
        #
        # The cost is weaker decorrelation over the first ~2*shuffle_buffer
        # samples of an epoch. `sample_stride` is the cheaper primary defence
        # against correlated windows — it decorrelates without buffering and
        # cuts I/O at the same time.
        shard_index, _ = _shard_info()
        rng    = random.Random(hash((self.epoch, shard_index, self.split)) & 0xFFFFFFFF)
        buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        taken  = 0

        for item in stream:
            if len(buffer) < self.shuffle_buffer:
                buffer.append(item)
                taken += 1
                # Grow phase: hold the first `shuffle_min_fill` samples back so
                # there is something to shuffle over, then emit on every second
                # intake (net +1 buffered per 2 fetched).
                if len(buffer) < self.shuffle_min_fill or taken % 2:
                    continue
                yield buffer.pop(rng.randrange(len(buffer)))
                continue

            # Steady state: swap a random buffered sample out for the new one.
            idx = rng.randrange(len(buffer))
            buffer[idx], item = item, buffer[idx]
            yield item

        rng.shuffle(buffer)
        yield from buffer

    def log_epoch_summary(self) -> None:
        logger.info(
            "[%s] yielded=%d  skipped: night-hour=%d nan=%d fetch=%d shape=%d | "
            "gap frames interpolated=%d | %s",
            self.split, self.n_yielded, self.n_skipped_hour, self.n_skipped_nan,
            self.n_skipped_error, self.n_skipped_shape, self.n_interpolated,
            self.cache.summary(),
        )
        attempted = self.n_yielded + self.n_skipped_nan + self.n_skipped_error
        if attempted and self.n_yielded < 0.5 * attempted:
            logger.warning(
                "[%s] only %d of %d attempted anchors produced a sample (%.0f%%). "
                "Every failure still costs an archive lookup, so this is where "
                "the wall-clock is going.",
                self.split, self.n_yielded, attempted,
                100 * self.n_yielded / attempted,
            )


# ---------------------------------------------------------------------------
# Public entry point called by Trainer.setup_data()
# ---------------------------------------------------------------------------

def build_dataloader(
    split:      str,
    config:     dict,
    shuffle:    bool = False,
    drop_last:  Optional[bool] = None,
    batch_size: Optional[int] = None,
    prime_mode: bool = False,
) -> DataLoader:
    """
    Build a DataLoader for the given split using the pyearthtools pipeline.

    Args:
        split:   "train" or "val"
        config:  full config dict from load_config()
        shuffle:   enable the in-stream reservoir shuffle (train only)
        drop_last: override dataloader.drop_last. Validation passes False so a
                   small split cannot lose every batch it has.
        prime_mode: populate the cache without decoding frames. Yields
                   placeholders, so the caller must not use the tensors.
        batch_size: override dataloader.batch_size. Cache priming passes a small
                   value: it never looks at the collated tensors, and a large
                   batch makes every worker hold that many assembled samples
                   (~10 MB each) before emitting anything.

    Returns:
        A DataLoader yielding (context, forecast) tensor pairs of shape
        (B, context_length, C, H, W) and (B, forecast_length, C, H, W).
    """
    dataset    = PipelineDataset(split, config, shuffle=shuffle,
                                 prime_mode=prime_mode)
    loader_cfg = config.get("dataloader", {})

    num_workers = int(loader_cfg.get("num_workers", 0))
    kwargs: dict = {}
    if num_workers > 0:
        kwargs["prefetch_factor"]    = int(loader_cfg.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = bool(loader_cfg.get("persistent_workers", True))

        # Workers are spawned rather than forked. This is a precaution, not a
        # fix for any observed hang: `_build_pipeline` is documented as not
        # fork-safe, and by the time this runs CUDA (and under DDP, NCCL) are
        # already live in the parent, so a forked worker would inherit state
        # neither is safe to inherit. `spawn` gives each worker a clean
        # interpreter instead. It costs a slower worker startup (pyearthtools,
        # dask and xarray are re-imported per worker), which
        # `persistent_workers=True` pays once per run rather than per epoch.
        kwargs["multiprocessing_context"] = loader_cfg.get(
            "multiprocessing_context", "spawn"
        )

    return DataLoader(
        dataset,
        batch_size  = (int(loader_cfg.get("batch_size", 8))
                       if batch_size is None else int(batch_size)),
        num_workers = num_workers,
        pin_memory  = loader_cfg.get("pin_memory", True),
        drop_last   = bool(loader_cfg.get("drop_last", True))
                      if drop_last is None else bool(drop_last),
        **kwargs,
    )
