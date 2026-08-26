"""Inference helpers for generating satellite image forecasts."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.models import LatentDiffusionTransformer
from src.training.config import TrainingConfig

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    """Return a usable torch device, falling back to CPU when CUDA is unavailable."""
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        return "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Requested GPU device %s is unavailable; falling back to CPU", device)
        return "cpu"
    return device


class Forecaster:
    """Inference wrapper for satellite image forecasting."""

    def __init__(self, model: LatentDiffusionTransformer, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        self.model.eval()

    def load_checkpoint(self, checkpoint_path: str):
        logger.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Checkpoint loaded successfully")

    @torch.no_grad()
    def forecast(self, context_images: torch.Tensor, num_steps: int = 6, num_samples: int = 1) -> torch.Tensor:
        context_images = context_images.to(self.device)
        samples = self.model.sample(
            context_images,
            num_forecast_steps=num_steps,
            num_samples=num_samples,
            ddim_steps=50,
            eta=0.0,
        )
        return samples.cpu()

    @torch.no_grad()
    def forecast_deterministic(self, context_images: torch.Tensor, num_steps: int = 6) -> torch.Tensor:
        context_images = context_images.to(self.device)
        forecast = self.model.forecast_deterministic(context_images, num_forecast_steps=num_steps)
        return forecast.cpu()

    def forecast_sequence(self, context_images: torch.Tensor, num_steps: int = 6, autoregressive: bool = False) -> torch.Tensor:
        all_frames = [context_images]
        if autoregressive:
            current_context = context_images
            while current_context.shape[1] < context_images.shape[1] + num_steps:
                forecast = self.forecast_deterministic(current_context, num_steps=min(6, context_images.shape[1] + num_steps - current_context.shape[1]))
                all_frames.append(forecast)
                current_context = torch.cat([current_context, forecast], dim=1)
        else:
            forecast = self.forecast_deterministic(context_images, num_steps)
            all_frames.append(forecast)
        return torch.cat(all_frames, dim=1)


class BatchPredictor:
    """Batch prediction handler for multiple images."""

    def __init__(self, forecaster: Forecaster, batch_size: int = 32):
        self.forecaster = forecaster
        self.batch_size = batch_size

    def predict_batch(self, data_loader, num_forecast_steps: int = 6, return_context: bool = True) -> List[np.ndarray]:
        predictions = []
        for context, _ in data_loader:
            pred = self.forecaster.forecast_deterministic(context, num_forecast_steps)
            if return_context:
                full_sequence = torch.cat([context, pred], dim=1)
                predictions.append(full_sequence.numpy())
            else:
                predictions.append(pred.numpy())
        return predictions


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda") -> Tuple[LatentDiffusionTransformer, Forecaster]:
    device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TrainingConfig.from_dict(checkpoint.get("config", {}))
    model = LatentDiffusionTransformer(
        image_channels=config.image_channels,
        image_size=config.image_size,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        num_transformer_layers=config.num_transformer_layers,
        num_heads=config.num_heads,
        feedforward_dim=config.feedforward_dim,
        num_diffusion_steps=config.num_diffusion_steps,
        denoiser_hidden_dim=config.denoiser_hidden_dim,
        dropout=config.dropout,
    )
    forecaster = Forecaster(model, checkpoint_path=checkpoint_path, device=device)
    return model, forecaster
