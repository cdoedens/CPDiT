"""Training configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

_REQUIRED_SECTIONS = ("model", "data", "training", "optimiser", "logging")

_REQUIRED_KEYS = (
    ("model", "image_channels"),
    ("model", "image_size"),
    ("model", "latent_channels"),
    ("model", "hidden_dim"),
    ("data",  "n_prior_sat"),
    ("data",  "n_post"),
    ("data",  "stats_path"),
    ("data",  "splits"),
    ("training", "checkpoint_dir"),
    ("training", "log_dir"),
)


def load_config(yaml_path: str | Path) -> Dict[str, Any]:
    """
    Load and validate a training config YAML.

    Returns the raw nested dict. All downstream code accesses it via
    config["section"]["key"] — no dataclass translation layer.

    Raises:
        FileNotFoundError: if the YAML file (or the stats file it names) does
            not exist.
        KeyError: if a required section or key is missing.
        ValueError: if the model geometry is inconsistent.
    """
    config = _load_merged(Path(yaml_path))
    _validate(config)
    return config


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


def _validate(config: Dict[str, Any]) -> None:
    for section in _REQUIRED_SECTIONS:
        if section not in config:
            raise KeyError(
                f"Missing required section '{section}' in config. "
                f"Required sections: {_REQUIRED_SECTIONS}"
            )

    for section, key in _REQUIRED_KEYS:
        if key not in config[section]:
            raise KeyError(f"Missing required config key '{section}.{key}'.")

    model_cfg     = config["model"]
    diffusion_cfg = model_cfg.get("diffusion", {})

    image_size = int(model_cfg["image_size"])
    if image_size % 8 != 0:
        raise ValueError(f"model.image_size must be divisible by 8, got {image_size}.")

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

    stats_path = Path(config["data"]["stats_path"])
    if not stats_path.exists():
        raise FileNotFoundError(
            f"data.stats_path does not exist: {stats_path}\n"
            f"Run scripts/calc_norm_stats.py first."
        )
