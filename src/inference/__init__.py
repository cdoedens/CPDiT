"""Inference helpers for generating satellite image forecasts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.models import LatentDiffusionTransformer

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    """Return a usable torch device, falling back to CPU when CUDA is unavailable."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Requested device %s is unavailable; falling back to CPU", device)
        return "cpu"
    return device


def build_model_from_config(config: dict) -> LatentDiffusionTransformer:
    """
    Construct a model from a training config dict.

    Kept in one place so training and inference cannot drift apart in how they
    map config keys onto constructor arguments.
    """
    model_cfg       = config["model"]
    transformer_cfg = model_cfg.get("transformer", {})
    diffusion_cfg   = model_cfg.get("diffusion", {})
    data_cfg        = config["data"]

    return LatentDiffusionTransformer(
        image_channels         = model_cfg["image_channels"],
        image_size             = model_cfg["image_size"],
        latent_channels        = model_cfg["latent_channels"],
        vae_hidden_dim         = model_cfg["hidden_dim"],
        context_length         = data_cfg["n_prior_sat"],
        max_forecast_steps     = model_cfg.get("max_forecast_steps", 32),
        num_transformer_layers = transformer_cfg["num_layers"],
        num_heads              = transformer_cfg["num_heads"],
        feedforward_dim        = transformer_cfg["feedforward_dim"],
        transformer_dim        = transformer_cfg.get("transformer_dim", 512),
        max_seq_len            = transformer_cfg.get("max_seq_len", 64),
        dropout                = transformer_cfg.get("dropout", 0.1),
        patch_size             = diffusion_cfg.get("patch_size", 4),
        denoiser_embed_dim     = diffusion_cfg.get("embed_dim", 768),
        denoiser_depth         = diffusion_cfg.get("num_blocks", 16),
        denoiser_heads         = diffusion_cfg.get("num_heads", 12),
        window_size            = diffusion_cfg.get("window_size", 31),
        ffn_mult               = diffusion_cfg.get("ffn_mult", 4),
        cond_dim               = diffusion_cfg.get("cond_dim", 256),
        use_natten             = diffusion_cfg.get("use_natten", None),
        sde                    = diffusion_cfg.get("sde", "vp_cosine"),
        beta_min               = diffusion_cfg.get("beta_min", 0.1),
        beta_max               = diffusion_cfg.get("beta_max", 20.0),
        cosine_s               = diffusion_cfg.get("cosine_s", 0.008),
        sigma_min              = diffusion_cfg.get("sigma_min", 0.01),
        sigma_max              = diffusion_cfg.get("sigma_max", 50.0),
        t_eps                  = diffusion_cfg.get("t_eps", None),
        loss_weighting         = diffusion_cfg.get("loss_weighting", "sigma2"),
        time_scale             = diffusion_cfg.get("time_scale", 1000.0),
        latent_scale           = model_cfg.get("latent_scale", None),
        latent_scale_momentum  = model_cfg.get("latent_scale_momentum", 0.99),
    )


class Forecaster:
    """Inference wrapper for satellite image forecasting."""

    def __init__(
        self,
        model: LatentDiffusionTransformer,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        num_steps: int = 100,
        sampler: str = "pc",
    ):
        self.device    = resolve_device(device)
        self.model     = model.to(self.device)
        self.num_steps = num_steps
        self.sampler   = sampler
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        self.model.eval()

    def load_checkpoint(self, checkpoint_path: str) -> None:
        logger.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Checkpoint loaded (epoch %s)", checkpoint.get("epoch", "?"))

    @torch.no_grad()
    def forecast(
        self,
        context_images: torch.Tensor,
        num_steps: int = 1,
        num_samples: int = 1,
        sampler: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Probabilistic forecast via the reverse-time SDE.

        Ensembles need the stochastic ``"pc"`` sampler: the probability-flow ODE
        is deterministic given its initial noise, so an "ensemble" from it only
        varies through that draw.

        Returns:
            (B * num_samples, num_steps, C, H, W)
        """
        context_images = context_images.to(self.device)
        samples = self.model.sample(
            context_images,
            num_forecast_steps = num_steps,
            num_samples        = num_samples,
            sampler            = sampler or self.sampler,
            num_steps          = self.num_steps,
        )
        return samples.cpu()

    @torch.no_grad()
    def forecast_deterministic(
        self, context_images: torch.Tensor, num_steps: int = 1
    ) -> torch.Tensor:
        context_images = context_images.to(self.device)
        forecast = self.model.forecast_deterministic(
            context_images, num_forecast_steps=num_steps, num_steps=self.num_steps
        )
        return forecast.cpu()

    @torch.no_grad()
    def forecast_sequence(
        self,
        context_images: torch.Tensor,
        num_steps: int,
        autoregressive: bool = False,
    ) -> torch.Tensor:
        """
        Forecast `num_steps` frames beyond the context.

        With ``autoregressive=True`` the model is rolled forward by feeding its
        own predictions back in, keeping the context window at its trained
        length. Errors compound, so prefer a direct multi-frame forecast when
        ``num_steps`` fits within the model's trained horizon.

        Returns:
            (B, context_length + num_steps, C, H, W)
        """
        context_length = self.model.context_length
        frames = [context_images]

        if not autoregressive:
            frames.append(self.forecast_deterministic(context_images, num_steps))
            return torch.cat(frames, dim=1)

        window    = context_images
        remaining = num_steps
        max_chunk = self.model.max_forecast_steps

        while remaining > 0:
            chunk    = min(max_chunk, remaining)
            forecast = self.forecast_deterministic(window, num_steps=chunk)
            frames.append(forecast)
            # Slide the context window forward, keeping its trained length.
            window    = torch.cat([window, forecast.to(window.device)], dim=1)[:, -context_length:]
            remaining -= chunk

        return torch.cat(frames, dim=1)


class BatchPredictor:
    """Batch prediction handler over a dataloader."""

    def __init__(self, forecaster: Forecaster):
        self.forecaster = forecaster

    def predict_batch(
        self,
        data_loader,
        num_forecast_steps: int = 1,
        return_context: bool = True,
    ) -> List[np.ndarray]:
        predictions = []
        for context, _ in data_loader:
            pred = self.forecaster.forecast_deterministic(context, num_forecast_steps)
            if return_context:
                predictions.append(torch.cat([context, pred], dim=1).numpy())
            else:
                predictions.append(pred.numpy())
        return predictions


def load_model_from_checkpoint(
    checkpoint_path: str | Path, device: str = "cuda"
) -> Tuple[LatentDiffusionTransformer, Forecaster]:
    """
    Rebuild a model from the config stored inside its checkpoint and return it
    alongside a ready-to-use Forecaster.
    """
    device     = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config")
    if not config:
        raise KeyError(
            f"Checkpoint {checkpoint_path} has no embedded 'config'; cannot "
            f"reconstruct the model architecture."
        )

    model = build_model_from_config(config)
    inf_cfg = config["model"].get("inference", {})

    forecaster = Forecaster(
        model, checkpoint_path=str(checkpoint_path), device=device,
        num_steps=inf_cfg.get("num_steps", 100),
        sampler=inf_cfg.get("sampler", "pc"),
    )
    return model, forecaster
