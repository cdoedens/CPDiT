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
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    _validate(config)
    return config


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
