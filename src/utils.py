"""Utility functions for training and evaluation"""

import torch
import numpy as np
from typing import Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: (batch, time, channels, height, width)
        targets: Same shape as predictions
        
    Returns:
        Dictionary of metrics
    """
    # Flatten spatial dimensions
    pred_flat = predictions.reshape(predictions.shape[0], predictions.shape[1], -1).mean(axis=2)
    targ_flat = targets.reshape(targets.shape[0], targets.shape[1], -1).mean(axis=2)
    
    mse = mean_squared_error(targ_flat, pred_flat)
    mae = mean_absolute_error(targ_flat, pred_flat)
    rmse = np.sqrt(mse)
    
    # Per-timestep metrics
    per_step_rmse = [np.sqrt(mean_squared_error(targ_flat[:, t], pred_flat[:, t])) 
                     for t in range(targ_flat.shape[1])]
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'per_step_rmse': per_step_rmse,
    }


def inverse_normalize(data: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """Inverse normalization to original scale"""
    return data * (max_val - min_val) + min_val


def crop_to_size(data: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Crop data to specified size (center crop)"""
    h, w = data.shape[-2:]
    h_crop, w_crop = size
    
    h_start = (h - h_crop) // 2
    w_start = (w - w_crop) // 2
    
    return data[..., h_start:h_start + h_crop, w_start:w_start + w_crop]


def temporal_smoothing(data: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply temporal smoothing"""
    from scipy.ndimage import uniform_filter1d
    
    return uniform_filter1d(data, size=kernel_size, axis=1)


def get_gradient(tensor: torch.Tensor) -> torch.Tensor:
    """Compute spatial gradients"""
    dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
    
    # Pad to same size
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))
    
    return torch.sqrt(dy**2 + dx**2)
