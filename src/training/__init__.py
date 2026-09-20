"""Training module."""

from .config import load_config

__all__ = ["load_config", "Trainer"]


def __getattr__(name: str):
    """
    Import `Trainer` lazily (PEP 562).

    `Trainer` pulls in `src.petdata`, and therefore the whole pyearthtools
    runtime, which only exists inside the HPC module environment. Loading and
    validating a config needs none of that, so importing it eagerly here made
    `from src.training.config import load_config` fail anywhere else — including
    in the tests that check the config guards.
    """
    if name == "Trainer":
        from .train import Trainer
        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
