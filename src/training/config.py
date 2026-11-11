"""Training configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

_REQUIRED_SECTIONS = ("model", "data", "training", "optimiser", "logging")

_REQUIRED_KEYS = (
    ("model", "image_size"),
    ("model", "latent_channels"),
    ("model", "hidden_dim"),
    ("data",  "n_prior_sat"),
    ("data",  "n_post"),
    ("data",  "stats_path"),
    ("data",  "splits"),
)

# The keys an inputs file may set. Model variants differ ONLY in what they are
# fed; anything else in an inputs file would make two "variants" differ in ways
# their names do not say, so it is rejected rather than merged.
INPUT_KEYS = ("himawari_vars", "himawari_bands", "barra_vars")


def channel_names(config: dict) -> list[str]:
    """
    The channel order of every frame the data pipeline produces, as variable
    names.

    Satellite variables first, then raw Himawari bands, then BARRA, each in the
    order the config lists them -- so `data.irradiance_channel` (default 0) is
    `himawari_vars[0]`. Bands are named by their band id (e.g. ``OBS_B13``),
    which is also their key in the statistics file.

    This is the single definition of that order. It is NOT the order the
    archive hands the variables back in: the Himawari accessor's `data_vars`
    order varies between fetches of the very same timestamp (MEASURED: two runs
    over 20240101T0000 gave ['surface_global_irradiance', 'solar_elevation']
    and the reverse), so stacking in `data_vars` order silently permuted the
    channels of an arbitrary subset of frames.
    """
    data_cfg = config["data"]
    return (list(data_cfg.get("himawari_vars", ["surface_global_irradiance"]))
            + list(data_cfg.get("himawari_bands", []))
            + list(data_cfg.get("barra_vars", ["RH24mean"])))


# Pixel pitch of the Himawari solar (p1s) grid, in degrees. The grid is also
# the model grid: BARRA and the raw bands are regridded onto it.
DEFAULT_GRID_STEP = 0.02

# Degrees added on every side of `data.region` when fetching from the archives.
# BARRA (AUST-11, 0.11 deg) is regridded onto the satellite pixels by nearest
# neighbour, which returns NaN for any pixel beyond the outermost BARRA cell
# centre. Bounding BARRA to the region itself therefore leaves a strip of NaN
# along the edges; fetching a margin around it and selecting the region after
# the regrid does not. One BARRA cell would do; 0.5 deg leaves room.
DEFAULT_FETCH_MARGIN = 0.5


def model_region(config: dict) -> tuple[float, float, float, float]:
    """
    ``data.region`` as ``(lat_min, lat_max, lon_min, lon_max)``.

    The values are the centres of the outermost pixels, inclusive, so the
    region is exactly the frame the model sees -- no further cropping.
    """
    data_cfg = config["data"]
    if "region" not in data_cfg:
        raise KeyError(
            "data.region is not set. Give the model domain as [lat_min, lat_max, "
            "lon_min, lon_max]: the centres of the outermost pixels, inclusive."
        )
    region = data_cfg["region"]
    if len(region) != 4:
        raise ValueError(f"data.region must be [lat_min, lat_max, lon_min, lon_max], got {region}.")
    lat_min, lat_max, lon_min, lon_max = (float(v) for v in region)
    if not (lat_min < lat_max and lon_min < lon_max):
        raise ValueError(f"data.region must have min < max on both axes, got {region}.")
    return lat_min, lat_max, lon_min, lon_max


def region_shape(config: dict) -> tuple[int, int]:
    """(n_lat, n_lon) pixels `data.region` spans on the `data.grid_step` grid."""
    lat_min, lat_max, lon_min, lon_max = model_region(config)
    step = float(config["data"].get("grid_step", DEFAULT_GRID_STEP))
    return (int(round((lat_max - lat_min) / step)) + 1,
            int(round((lon_max - lon_min) / step)) + 1)


def fetch_bounds(config: dict) -> list[float]:
    """`data.region` grown by `data.fetch_margin` on every side, for `Bounding`."""
    lat_min, lat_max, lon_min, lon_max = model_region(config)
    m = float(config["data"].get("fetch_margin", DEFAULT_FETCH_MARGIN))
    return [lat_min - m, lat_max + m, lon_min - m, lon_max + m]


def load_config(
    yaml_path: str | Path,
    inputs:    str | Path | None = None,
    require_stats: bool = True,
) -> Dict[str, Any]:
    """
    Load and validate a training config YAML.

    ``inputs`` names an inputs file (configs/inputs/*.yaml) that sets the
    input variables of one model variant on top of the stage config. Its file
    stem is the variant name, recorded under ``config["run"]["variant"]`` and
    used for the run directory. ``model.image_channels`` is derived from the
    resulting channel list, so it never has to be kept in step by hand.

    ``require_stats=False`` skips the check that the statistics file exists
    and covers every channel, for the script that computes it.

    Returns the raw nested dict. All downstream code accesses it via
    config["section"]["key"] — no dataclass translation layer.

    Raises:
        FileNotFoundError: if the YAML file (or the stats file it names) does
            not exist.
        KeyError: if a required section or key is missing.
        ValueError: if the model geometry is inconsistent.
    """
    config = _load_merged(Path(yaml_path))
    run = config.setdefault("run", {})
    run["config"] = str(Path(yaml_path).resolve())
    if inputs is not None:
        _apply_inputs(config, Path(inputs))
    _resolve_image_channels(config)
    _validate(config, require_stats=require_stats)
    return config


def _apply_inputs(config: Dict[str, Any], path: Path) -> None:
    """Overlay an inputs file's data keys and record the variant it names."""
    if not path.exists():
        raise FileNotFoundError(f"Inputs file not found: {path}")
    with open(path) as f:
        overlay = yaml.safe_load(f) or {}

    data = overlay.pop("data", {}) or {}
    stray = sorted(overlay) + [f"data.{k}" for k in sorted(data) if k not in INPUT_KEYS]
    if stray:
        raise KeyError(
            f"Inputs file {path} sets {stray}; an inputs file may only set "
            f"{', '.join('data.' + k for k in INPUT_KEYS)}. Put shared settings "
            f"in the stage config instead."
        )
    if not data:
        raise KeyError(f"Inputs file {path} sets no input variables.")

    # Replace, not merge: an inputs file states the complete input set.
    for key in INPUT_KEYS:
        config["data"][key] = list(data.get(key) or [])
    config["run"]["variant"] = path.stem
    config["run"]["inputs"] = str(path.resolve())


def _resolve_image_channels(config: Dict[str, Any]) -> None:
    """
    Derive model.image_channels from the input variables.

    A config may still state it, but a stated value that disagrees with the
    inputs is an error: the VAE's first conv would be sized for one input set
    and fed another.
    """
    data = config.get("data", {})
    if not any(k in data for k in INPUT_KEYS):
        return                      # nothing to derive from; _validate decides
    n = len(channel_names(config))
    model_cfg = config.setdefault("model", {})
    stated = model_cfg.get("image_channels")
    if stated is not None and int(stated) != n:
        raise ValueError(
            f"model.image_channels is {stated} but the inputs give {n} channels: "
            f"{channel_names(config)}. Remove model.image_channels from the "
            f"config; it is derived from the inputs."
        )
    model_cfg["image_channels"] = n


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively overlay `override` onto `base`, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_merged(path: Path, _seen: tuple[Path, ...] = ()) -> Dict[str, Any]:
    """
    Read a config, applying an optional ``extends:`` base first.

    The three training stages differ in a handful of keys but share the whole
    data pipeline, model geometry and HPC block. Without inheritance each stage
    would need its own full copy of the file, and the copies would drift — which
    for something like `data.splits` or `stats_path` means silently training on
    the wrong thing. A stage config therefore names its base and overrides only
    what it actually changes.

    ``extends`` is resolved relative to the file that names it.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    resolved = path.resolve()
    if resolved in _seen:
        chain = " -> ".join(str(p) for p in (*_seen, resolved))
        raise ValueError(f"Circular 'extends' in config files: {chain}")

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    base_name = config.pop("extends", None)
    if base_name is None:
        return config

    base_path = (path.parent / str(base_name)).resolve()
    base = _load_merged(base_path, (*_seen, resolved))
    return _deep_merge(base, config)


def _validate_region(config: Dict[str, Any], image_size: int) -> None:
    """
    The spatial domain: `data.region` must span exactly image_size pixels.

    Checked here, before any data is touched, because the dataset can only
    discover a mismatch frame by frame -- and it records each mismatched frame
    as unusable in the cache, so a typo in the region would quietly poison a
    whole namespace.
    """
    data_cfg = config["data"]
    legacy = [k for k in ("bounds", "crop_offset") if k in data_cfg]
    if legacy:
        raise KeyError(
            f"data.{' and data.'.join(legacy)} {'is' if len(legacy) == 1 else 'are'} no "
            "longer used. Set data.region to the model domain itself "
            "([lat_min, lat_max, lon_min, lon_max], outermost pixel centres, "
            "inclusive); the pipeline fetches data.fetch_margin degrees around it "
            "and selects the region after regridding BARRA."
        )
    if "region" not in data_cfg:
        return                      # the dataset insists on it when it is built
    shape = region_shape(config)
    if shape != (image_size, image_size):
        step = float(data_cfg.get("grid_step", DEFAULT_GRID_STEP))
        raise ValueError(
            f"data.region {data_cfg['region']} spans {shape[0]} x {shape[1]} pixels "
            f"at grid_step {step}, but model.image_size is {image_size}. Each "
            f"side must cover (image_size - 1) * grid_step = "
            f"{(image_size - 1) * step:.4g} degrees between the outermost pixel centres."
        )
    margin = float(data_cfg.get("fetch_margin", DEFAULT_FETCH_MARGIN))
    if margin < 0:
        raise ValueError(f"data.fetch_margin must be >= 0, got {margin}.")


def _validate(config: Dict[str, Any], require_stats: bool = True) -> None:
    for section in _REQUIRED_SECTIONS:
        if section not in config:
            raise KeyError(
                f"Missing required section '{section}' in config. "
                f"Required sections: {_REQUIRED_SECTIONS}"
            )

    for section, key in _REQUIRED_KEYS:
        if key not in config[section]:
            raise KeyError(f"Missing required config key '{section}.{key}'.")

    if not any(k in config["data"] for k in INPUT_KEYS):
        raise KeyError(
            "No input variables (data.himawari_vars / himawari_bands / "
            "barra_vars). Pass an inputs file, e.g. "
            "--inputs configs/inputs/baseline.yaml."
        )
    if "image_channels" not in config["model"]:
        raise KeyError("Missing required config key 'model.image_channels'.")

    training_cfg = config["training"]
    if not (training_cfg.get("runs_root") or training_cfg.get("checkpoint_dir")):
        raise KeyError(
            "Set training.runs_root (one directory per run, recommended) or "
            "training.checkpoint_dir (a fixed directory)."
        )

    model_cfg     = config["model"]
    diffusion_cfg = model_cfg.get("diffusion", {})

    image_size = int(model_cfg["image_size"])
    if image_size % 8 != 0:
        raise ValueError(f"model.image_size must be divisible by 8, got {image_size}.")

    _validate_region(config, image_size)

    # The VAE downsamples by 8; the DiT then patchifies the latent map.
    latent_size = image_size // 8
    patch_size  = int(diffusion_cfg.get("patch_size", 4))
    if latent_size % patch_size != 0:
        raise ValueError(
            f"Latent size {latent_size} (= image_size / 8) must be divisible by "
            f"model.diffusion.patch_size {patch_size}."
        )

    embed_dim = int(diffusion_cfg.get("embed_dim", 768))
    heads     = int(diffusion_cfg.get("num_heads", 12))
    if embed_dim % heads != 0:
        raise ValueError(
            f"model.diffusion.embed_dim {embed_dim} must be divisible by "
            f"model.diffusion.num_heads {heads}."
        )
    if embed_dim % 4 != 0:
        raise ValueError(
            f"model.diffusion.embed_dim {embed_dim} must be divisible by 4 for "
            f"the 2-D sin-cos positional embedding."
        )

    window = int(diffusion_cfg.get("window_size", 31))
    if window % 2 == 0:
        raise ValueError(
            f"model.diffusion.window_size must be odd (neighbourhood windows are "
            f"centred on the query), got {window}."
        )

    sde_name = str(diffusion_cfg.get("sde", "vp_cosine")).lower()
    if sde_name not in ("vp", "vp_cosine", "ve"):
        raise ValueError(
            f"model.diffusion.sde must be one of vp, vp_cosine, ve; got {sde_name!r}."
        )

    weighting = str(diffusion_cfg.get("loss_weighting", "sigma2")).lower()
    if weighting not in ("sigma2", "likelihood"):
        raise ValueError(
            f"model.diffusion.loss_weighting must be 'sigma2' or 'likelihood'; "
            f"got {weighting!r}."
        )

    latent_norm = str(model_cfg.get("latent_norm", "ema")).lower()
    if latent_norm not in ("ema", "batch"):
        raise ValueError(
            f"model.latent_norm must be 'ema' or 'batch'; got {latent_norm!r}."
        )

    # The guard that the first real training run needed and did not have.
    # Under 'ema' the scaling denominator is a constant as far as autograd is
    # concerned, so the diffusion loss is not scale-invariant and shrinking the
    # latent towards zero is its global minimum. Letting that gradient reach the
    # encoder destroys the latent space while every training number improves.
    training_cfg = config.get("training", {})
    if not training_cfg.get("detach_latents", True) and latent_norm != "batch":
        raise ValueError(
            "training.detach_latents is false but model.latent_norm is 'ema'. "
            "That combination makes collapsing the latent the global minimum of "
            "the diffusion loss: the loss falls while forecast skill degrades. "
            "Set model.latent_norm: batch to make the loss scale-invariant "
            "before letting the diffusion gradient reach the encoder."
        )

    for key in ("vae_loss_weight", "diffusion_loss_weight", "pixel_loss_weight"):
        value = training_cfg.get(key)
        if value is not None and float(value) < 0:
            raise ValueError(f"training.{key} must be >= 0; got {value}.")

    if float(training_cfg.get("pixel_loss_weight", 0.0)) > 0:
        max_t = float(training_cfg.get("pixel_loss_max_t", 0.3))
        if not 0.0 < max_t <= 1.0:
            raise ValueError(
                f"training.pixel_loss_max_t must be in (0, 1]; got {max_t}. "
                "Above ~0.3 the 1/alpha factor in Tweedie amplifies score error "
                "so much that the decoded-x0 gradient is noise."
            )

    monitor = str(training_cfg.get("early_stopping", {}).get("monitor") or "")
    if monitor.startswith("forecast_"):
        if int(config.get("evaluation", {}).get("forecast_eval_batches", 0)) <= 0:
            raise ValueError(
                f"training.early_stopping.monitor is {monitor!r}, but "
                "evaluation.forecast_eval_batches is 0, so that metric is never "
                "computed and early stopping would never see a value."
            )

    # A monitor the active stage holds at a constant zero is worse than no
    # monitor: best-model selection silently freezes on the first epoch and
    # early stopping never fires, while the log still prints a "new best" line.
    vae_off  = float(training_cfg.get("vae_loss_weight", 0.1)) == 0
    diff_off = float(training_cfg.get("diffusion_loss_weight", 1.0)) == 0
    dead = (
        ({"recon_loss", "vae_loss", "kl_loss"} if vae_off else set())
        | ({"diff_loss", "forecast_rmse", "forecast_mse", "forecast_mae",
            "irradiance_mse", "irradiance_mae"} if diff_off else set())
    )
    if monitor in dead:
        raise ValueError(
            f"training.early_stopping.monitor is {monitor!r}, but this stage has "
            f"{'vae_loss_weight' if monitor.endswith(('recon_loss', 'vae_loss', 'kl_loss')) else 'diffusion_loss_weight'}"
            " = 0, so that metric is a constant and best-model selection would "
            "freeze on the first epoch. Monitor forecast_rmse in a stage that "
            "runs the denoiser, or recon_loss in a VAE-only stage."
        )

    sampler = str(config.get("evaluation", {}).get("sampler", "ode")).lower()
    if sampler not in ("pc", "ode"):
        raise ValueError(f"evaluation.sampler must be 'pc' or 'ode'; got {sampler!r}.")

    n_post = int(config["data"]["n_post"])
    max_forecast = int(model_cfg.get("max_forecast_steps", 32))
    if n_post > max_forecast:
        raise ValueError(
            f"data.n_post {n_post} exceeds model.max_forecast_steps {max_forecast}."
        )

    latent_scale = model_cfg.get("latent_scale")
    if isinstance(latent_scale, str) and latent_scale != "auto":
        raise ValueError(
            f"model.latent_scale must be a number, null or 'auto'; got {latent_scale!r}."
        )

    if not require_stats:
        return
    stats_path = Path(config["data"]["stats_path"])
    if not stats_path.exists():
        raise FileNotFoundError(
            f"data.stats_path does not exist: {stats_path}\n"
            f"Run scripts/calc_norm_stats.py first."
        )

    # Caught here rather than in the dataloader, where a missing entry fails
    # every frame inside a worker and reads as a stream that yields nothing.
    import json

    with open(stats_path) as f:
        stats = json.load(f)
    missing = [v for v in channel_names(config) if v not in stats]
    if missing:
        raise KeyError(
            f"No normalisation statistics for {missing} in {stats_path}. Run: "
            f"python scripts/calc_norm_stats.py --config <cfg> --inputs <inputs>."
        )
