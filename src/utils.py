"""Utility functions for training and evaluation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    channel: int | None = 0,
) -> dict:
    """
    Compute evaluation metrics for a batch of forecasts.

    Args:
        predictions: (batch, time, channels, height, width)
        targets:     same shape as predictions
        channel:     score this channel only (None scores all channels).

    Returns:
        Dictionary of metrics. Note that `mse`/`mae`/`rmse` are computed
        per-pixel: averaging over space *before* scoring would only measure
        domain-mean error and would hide every spatial mistake the model makes.
        The domain-mean ("field average") errors are reported separately, since
        they are the quantity a site-level power forecast cares about.
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )

    if channel is not None:
        predictions = predictions[:, :, channel:channel + 1]
        targets     = targets[:, :, channel:channel + 1]

    error = predictions - targets

    # Per-pixel errors.
    mse  = float(np.mean(error ** 2))
    mae  = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))
    bias = float(np.mean(error))

    # Domain-mean (field average) errors.
    pred_mean = predictions.mean(axis=(2, 3, 4))
    targ_mean = targets.mean(axis=(2, 3, 4))
    field_err = pred_mean - targ_mean
    field_mse = float(np.mean(field_err ** 2))

    # Per-lead-time RMSE, the curve that actually matters for a nowcast.
    per_step_rmse = [
        float(np.sqrt(np.mean(error[:, t] ** 2))) for t in range(error.shape[1])
    ]
    per_step_mae = [
        float(np.mean(np.abs(error[:, t]))) for t in range(error.shape[1])
    ]

    return {
        "mse":            mse,
        "mae":            mae,
        "rmse":           rmse,
        "bias":           bias,
        "field_mean_mse":  field_mse,
        "field_mean_rmse": float(np.sqrt(field_mse)),
        "field_mean_mae":  float(np.mean(np.abs(field_err))),
        "per_step_rmse":  per_step_rmse,
        "per_step_mae":   per_step_mae,
    }


def persistence_baseline(context: np.ndarray, forecast_length: int) -> np.ndarray:
    """
    Persistence forecast: repeat the last context frame.

    Any nowcasting model must beat this to be worth running, so evaluation
    should always report it alongside the model.

    Args:
        context: (batch, context_length, channels, height, width)
    Returns:
        (batch, forecast_length, channels, height, width)
    """
    last = context[:, -1:]
    return np.repeat(last, forecast_length, axis=1)


def denormalise(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Invert the z-score applied by the dataloader, returning physical units."""
    return data * std + mean


def crop_to_size(data: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Centre-crop the trailing two dimensions to `size`."""
    h, w = data.shape[-2:]
    h_crop, w_crop = size
    h_start = (h - h_crop) // 2
    w_start = (w - w_crop) // 2
    return data[..., h_start:h_start + h_crop, w_start:w_start + w_crop]


def temporal_smoothing(data: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a uniform temporal filter along the time axis."""
    from scipy.ndimage import uniform_filter1d

    return uniform_filter1d(data, size=kernel_size, axis=1)


def get_gradient(tensor: torch.Tensor) -> torch.Tensor:
    """Spatial gradient magnitude over the trailing two dimensions."""
    dy = tensor[..., 1:, :] - tensor[..., :-1, :]
    dx = tensor[..., :, 1:] - tensor[..., :, :-1]

    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))

    return torch.sqrt(dy ** 2 + dx ** 2 + 1e-12)
