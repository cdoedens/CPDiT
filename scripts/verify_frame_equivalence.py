"""
Verify that per-frame assembly reproduces the old windowed pipeline exactly.

The dataloader used to fetch a whole 13-frame window per sample through
`TemporalWindow`; it now fetches and caches frames individually so each
timestamp is decoded once instead of ~4 times. That is only a safe change if
the assembled tensor is unchanged, which needs real archive data to confirm —
unit tests can check the window arithmetic but not the data itself.

This script rebuilds the old path inline and diffs it against the new one.
Last run: max abs diff 0.000e+00 (bit-identical) at 20200211T2300.

Usage (on a node with archive access):
    python scripts/verify_frame_equivalence.py
"""
import sys, functools, numpy as np, torch, xarray as xr
sys.path.insert(0, "/home/548/cd3022/repos/CPDiT")
import pyearthtools.data as petdata, pyearthtools.pipeline as petpipe
import importlib; importlib.import_module("site_archive_nci")
from src.training.config import load_config
import src.petdata as P

def old_sample(cfg, anchor):
    """The previous windowed path, reconstructed."""
    d = cfg["data"]; b = d["bounds"]; npr, npo = d["n_prior_sat"], d["n_post"]
    sat = petpipe.Pipeline(
        petdata.archive.Himawari(d["himawari_vars"]),
        petpipe.operations.xarray.Sort(order=['time','latitude','longitude']),
        petpipe.operations.xarray.AlignDataVariableDimensionsToDatasetCoords(),
        petdata.transform.region.Bounding(*b),
        petpipe.modifications.TemporalWindow(
            prior_indexes=list(range(-npr-1, 0)), posterior_indexes=list(range(0, npo+1)),
            merge_method=functools.partial(xr.concat, dim='time'),
            timedelta=petdata.time.TimeDelta((10, "minutes"))),
        petpipe.operations.xarray.Merge(),
        exceptions_to_ignore=petdata.exceptions.DataNotFoundError)
    bar = petpipe.Pipeline(
        petdata.archive.BARRA_V2(d["barra_vars"], domain_id=d["barra_domain"],
                                 frequency=d["barra_freq"]),
        petdata.transforms.coordinates.Drop("crs"),
        petpipe.operations.xarray.Sort(order=['time','latitude','longitude']),
        petpipe.operations.xarray.AlignDataVariableDimensionsToDatasetCoords(),
        petdata.transform.region.Bounding(*b),
        petpipe.modifications.TemporalWindow(
            prior_indexes=list(range(-3, 0)), posterior_indexes=list(range(0, 2)),
            merge_method=functools.partial(xr.concat, dim='time'),
            timedelta=petdata.time.TimeDelta((1, "h"))),
        petpipe.operations.xarray.Merge(),
        exceptions_to_ignore=petdata.exceptions.DataNotFoundError)

    ds = P.PipelineDataset("train", cfg, shuffle=False)
    s, bb = sat[anchor], bar[anchor]
    s = ds._normalise(s); bb = ds._normalise(bb)
    bi = bb.interp(latitude=s.latitude, longitude=s.longitude, time=s.time, method='nearest')
    comb = xr.merge([s, bi]).isel(time=slice(1, -1),
        latitude=slice(10, 10+256), longitude=slice(10, 10+256))
    arr = np.transpose(np.stack([comb[v].values for v in comb.data_vars], 0), (1,0,2,3))
    return torch.from_numpy(np.ascontiguousarray(arr)).float()

def main():
    cfg = load_config("/home/548/cd3022/repos/CPDiT/configs/train_config.yaml")
    cfg["dataloader"]["cache_dir"] = None
    anchor = "20200211T2300"

    old = old_sample(cfg, anchor)
    ds = P.PipelineDataset("train", cfg, shuffle=False)
    ds._sat_pipe, ds._bar_pipe = P._build_frame_pipelines(cfg)
    ds._bar_memo = {}
    new = ds._sample(anchor)

    print(f"old {tuple(old.shape)}  new {None if new is None else tuple(new.shape)}", flush=True)
    if new is None:
        print("NEW RETURNED None"); return
    same_shape = old.shape == new.shape
    maxdiff = float((old - new).abs().max()) if same_shape else float("nan")
    print(f"shapes match: {same_shape} | max abs diff: {maxdiff:.3e}", flush=True)
    print("EQUIVALENT" if same_shape and maxdiff < 1e-5 else "MISMATCH", flush=True)

if __name__ == "__main__":
    main()
