"""Training entry point for the latent diffusion transformer."""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_MLFLOW = False

from src.data import build_dataloader
from src.models import LatentDiffusionTransformer
from .config import TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Training loop manager for the staged LDM pipeline."""

    def __init__(self, config: TrainingConfig, device: str = "cuda"):
        self.config = config
        self.device = device

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        self.model = LatentDiffusionTransformer(
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
        ).to(device)

        if config.stage == 2 and config.freeze_vae:
            self.model.freeze_vae()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.mlflow_enabled = HAS_MLFLOW
        if self.mlflow_enabled:
            mlflow.set_tracking_uri(config.tracking_uri)
            mlflow.set_experiment(config.experiment_name)

    def setup_data(self) -> tuple[DataLoader, Optional[DataLoader]]:
        """Build training and validation data loaders from the YAML config."""
        logger.info("Setting up datasets...")
        cfg_dict = self.config.to_nested_dict()
        train_loader = build_dataloader("train", cfg_dict, shuffle=True)
        val_loader = build_dataloader("val", cfg_dict, shuffle=False)
        return train_loader, val_loader

    def _stage1_step(self, context: torch.Tensor, forecast: torch.Tensor) -> torch.Tensor:
        """Train the VAE alone on flattened image frames."""
        images = torch.cat([context, forecast], dim=1)
        flat_images = images.reshape(-1, *images.shape[2:])
        x_recon, mu, logvar = self.model.vae(flat_images)
        loss, _, _ = self.model.vae.vae_loss(flat_images, x_recon, mu, logvar, beta=0.001)
        return loss

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc="Training")

        for context, forecast in pbar:
            context = context.to(self.device)
            forecast = forecast.to(self.device)
            self.optimizer.zero_grad()

            if self.config.stage == 1:
                loss = self._stage1_step(context, forecast)
            else:
                loss, _ = self.model(context, forecast)

            loss.backward()
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.gradient_clip_norm)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        return total_loss / max(1, len(train_loader))

    def validate(self, val_loader: Optional[DataLoader]) -> Optional[float]:
        """Validate the current model."""
        if val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for context, forecast in tqdm(val_loader, desc="Validation"):
                context = context.to(self.device)
                forecast = forecast.to(self.device)

                if self.config.stage == 1:
                    loss = self._stage1_step(context, forecast)
                else:
                    loss, _ = self.model(context, forecast)

                total_loss += loss.item()

        return total_loss / max(1, len(val_loader))

    def save_checkpoint(self, epoch: int, val_loss: Optional[float] = None) -> Path:
        """Save a checkpoint containing weights and config."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "val_loss": val_loss,
        }
        path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
        return path

    def train(self, num_epochs: Optional[int] = None):
        """Run the full training loop."""
        num_epochs = num_epochs or self.config.max_epochs
        train_loader, val_loader = self.setup_data()

        if self.mlflow_enabled:
            with mlflow.start_run(run_name=self.config.run_name):
                mlflow.log_params(self.config.to_dict())
                self._training_loop(train_loader, val_loader, num_epochs)
        else:
            self._training_loop(train_loader, val_loader, num_epochs)

    def _training_loop(self, train_loader: DataLoader, val_loader: Optional[DataLoader], num_epochs: int):
        """Core loop with validation, checkpointing, and logging."""
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Train Loss: {train_loss:.6f}")

            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                logger.info(f"Val Loss: {val_loss:.6f}")

            if self.mlflow_enabled:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                if val_loss is not None:
                    mlflow.log_metric("val_loss", val_loss, step=epoch)

            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                saved_path = self.save_checkpoint(epoch + 1, val_loss)
                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    saved_path = self.save_checkpoint(epoch + 1, val_loss)
                    logger.info(f"New best model! Val Loss: {val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train the CPDiT latent diffusion transformer")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    config = TrainingConfig.from_yaml(args.config)
    trainer = Trainer(config, device=args.device)
    trainer.train()


if __name__ == "__main__":
    main()
