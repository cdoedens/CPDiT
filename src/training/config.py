"""Training configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(yaml_path: str | Path) -> Dict[str, Any]:
    """
    Load and validate a training config YAML.

    Returns the raw nested dict. All downstream code accesses it via
    config["section"]["key"] — no dataclass translation layer.

    Raises:
        FileNotFoundError: if the YAML file does not exist.
        KeyError: if a required top-level section is missing.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    _validate(config)
    return config


_REQUIRED_SECTIONS = ("model", "data", "training", "optimiser", "logging")

def _validate(config: Dict[str, Any]) -> None:
    for section in _REQUIRED_SECTIONS:
        if section not in config:
            raise KeyError(
                f"Missing required section '{section}' in config. "
                f"Required sections: {_REQUIRED_SECTIONS}"
            )
    qt_path = (
        config.get("data", {})
        .get("normalisation_stats", {})
        .get("barra_quantile_transforms")
    )
    if qt_path is not None and not Path(qt_path).exists():
        raise FileNotFoundError(
            f"barra_quantile_transforms path does not exist: {qt_path}\n"
            f"Run scripts/recompute_barra_stats.py first."
        )
