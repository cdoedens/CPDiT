"""Training entry point for the latent diffusion transformer."""

import argparse
import logging
import math
import os                                          # FIX 1: was missing entirely
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist                   # FIX 2: was missing entirely
from torch.nn.parallel import DistributedDataParallel as DDP  # FIX 3: was missing
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src_testing.data import build_dataloader
from src_testing.models import LatentDiffusionTransformer
from .config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Training loop manager for the staged LDM pipeline."""
    def __init__(self, config: dict, device: str = "cuda"):
        self.config     = config
        self.device     = device
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.is_primary = self.local_rank == 0
    
        model_cfg       = config["model"]
        training_cfg    = config["training"]
        optimiser_cfg   = config["optimiser"]
        transformer_cfg = model_cfg.get("transformer", {})
        diffusion_cfg   = model_cfg.get("diffusion", {})
    
        self.stage              = training_cfg["stage"]
        self.gradient_clip_norm = training_cfg.get("gradient_clip_norm", 1.0)
        self.checkpoint_dir     = Path(training_cfg["checkpoint_dir"])
        self.save_every         = training_cfg.get("save_every_n_epochs", 5)
        self.vae_beta           = training_cfg.get("vae_beta", 0.01)
        self.vae_ssim_weight    = training_cfg.get("vae_ssim_weight", 0.1)
    
        epoch_cfg       = training_cfg.get("max_epochs", {})
        self.max_epochs = (
            epoch_cfg.get(f"stage{self.stage}", 100)
            if isinstance(epoch_cfg, dict)
            else int(epoch_cfg)
        )
    
        stage_opt = optimiser_cfg.get(f"stage{self.stage}", {})
    
        if self.is_primary:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            Path(training_cfg["log_dir"]).mkdir(parents=True, exist_ok=True)
    
        # ── Build model ────────────────────────────────────────────────────
        self.model = LatentDiffusionTransformer(
            image_channels         = model_cfg["image_channels"],
            image_size             = model_cfg["image_size"],
            latent_channels        = model_cfg["latent_channels"],
            vae_hidden_dim         = model_cfg["hidden_dim"],
            num_transformer_layers = transformer_cfg["num_layers"],
            num_heads              = transformer_cfg["num_heads"],
            feedforward_dim        = transformer_cfg["feedforward_dim"],
            num_diffusion_steps    = diffusion_cfg["num_steps"],
            denoiser_hidden_dim    = diffusion_cfg["denoiser_hidden_dim"],
            dropout                = transformer_cfg.get("dropout", 0.1),
        ).to(device)
    
        # ── Stage 1 weights (must happen before freeze/DDP/compile) ───────
        if self.stage == 2:
            stage1_ckpt = training_cfg.get("stage1_checkpoint")
            if stage1_ckpt is None:
                raise ValueError(
                    "training.stage1_checkpoint must be set in config for Stage 2 training."
                )
            self._load_stage1_weights(stage1_ckpt)
    
        # ── Stage 2: freeze VAE, DDP-wrap denoiser, compile denoiser ──────
        # Order matters critically:
        #   1. freeze  →  2. DDP  →  3. compile
        # Compiling before DDP causes the int-symbol graph-split bug.
        # Compiling the full model (including frozen VAE) wastes compile
        # budget and triggers shape-symbol leakage across submod boundaries.
        if self.stage == 2:
            # 1. Freeze VAE so its parameters are excluded from DDP buckets
            #    and the optimiser.
            self.model.freeze_vae()
    
            # 2. Wrap only the denoiser in DDP.
            if dist.is_initialized():
                self.model.denoiser = DDP(
                    self.model.denoiser,
                    device_ids=[self.local_rank],
                )
    
            # 3. Compile only the denoiser (already DDP-wrapped).
            #    - dynamic=True  → handles variable batch/sequence shapes
            #    - optimize_ddp=False → stops Dynamo splitting the graph
            #      across DDP bucket boundaries (the source of the int-meta bug)
            #    - mode="reduce-overhead" is fine here; drop to "default" if
            #      you see further inductor issues
            if training_cfg.get("compile_model", True):
                if self.is_primary:
                    logger.info("Compiling denoiser with torch.compile...")
            
                # optimize_ddp is a Dynamo global config, not an inductor option —
                # it cannot be passed via torch.compile(options={...})
                torch._dynamo.config.optimize_ddp = False
            
                self.model.denoiser = torch.compile(
                    self.model.denoiser,
                    dynamic=True,
                    mode="reduce-overhead",
                )
    
        # ── Stage 1: compile + DDP the whole model ────────────────────────
        else:
            if training_cfg.get("compile_model", True):
                if self.is_primary:
                    logger.info("Compiling model with torch.compile...")
        
                torch._dynamo.config.optimize_ddp = False
        
                self.model = torch.compile(
                    self.model,
                    dynamic=True,
                    mode="reduce-overhead",
                )
        
            if dist.is_initialized():
                self.model = DDP(self.model, device_ids=[self.local_rank])
    
        # ── Optimiser: only non-frozen parameters ─────────────────────────
        # For stage 2 this naturally excludes the VAE since freeze_vae()
        # sets requires_grad=False on its parameters.
        betas = tuple(stage_opt.get("betas", [0.9, 0.999]))
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr           = stage_opt["lr"],
            weight_decay = stage_opt.get("weight_decay", 1e-4),
            betas        = betas,
        )
    
        sched_cfg     = optimiser_cfg.get("scheduler", {})
        warmup_epochs = sched_cfg.get("warmup_epochs", 0)
        min_lr        = sched_cfg.get("min_lr", 1e-6)
        base_lr       = stage_opt["lr"]
    
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, self.max_epochs - warmup_epochs)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine
    
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
    
        self.use_amp   = training_cfg.get("mixed_precision", False) and device.startswith("cuda")
        self.amp_dtype = torch.bfloat16
        self.scaler    = torch.amp.GradScaler("cuda", enabled=self.use_amp)
    
        logging_cfg       = config.get("logging", {})
        self.run_name     = logging_cfg.get("project_name", "baseline")
        self.mlflow_enabled = HAS_MLFLOW and self.is_primary
        if self.mlflow_enabled:
            mlflow.set_tracking_uri(config.get("tracking_uri", "http://localhost:5000"))
            mlflow.set_experiment(logging_cfg.get("experiment_name", "CPDiT"))


    # ------------------------------------------------------------------ #
    # VAE checkpoint                                                     #
    # ------------------------------------------------------------------ #
    def _load_stage1_weights(self, checkpoint_path: str | Path) -> None:
        """Load VAE weights from a Stage 1 checkpoint into the current model."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {path}")
    
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
    
        # Filter to VAE weights only
        vae_state = {
            k.removeprefix("vae."): v
            for k, v in state_dict.items()
            if k.startswith("vae.")
        }
    
        inner = self.model.module if isinstance(self.model, DDP) else self.model
        missing, unexpected = inner.vae.load_state_dict(vae_state, strict=True)
    
        if missing:
            raise RuntimeError(f"Missing keys loading VAE weights: {missing}")
    
        if self.is_primary:
            logger.info(f"Loaded Stage 1 VAE weights from {path}")


    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #

    def setup_data(self) -> tuple[DataLoader, Optional[DataLoader]]:
        if self.is_primary:
            logger.info("Setting up datasets...")
        train_loader = build_dataloader("train", self.config, shuffle=True)
        val_loader   = build_dataloader("val",   self.config, shuffle=False)
        return train_loader, val_loader

    # ------------------------------------------------------------------ #
    # Forward steps                                                        #
    # ------------------------------------------------------------------ #

    def _get_inner_model(self):
        """Return the unwrapped LatentDiffusionTransformer regardless of stage."""
        # Stage 1: self.model may be DDP(compile(LDT))
        # Stage 2: self.model is LDT directly (denoiser submodule is DDP-wrapped)
        return self.model.module if isinstance(self.model, DDP) else self.model
    
    def _get_vae(self):
        return self._get_inner_model().vae

    def _stage1_step(self, context: torch.Tensor, forecast: torch.Tensor) -> torch.Tensor:
        vae         = self._get_vae()
        images      = torch.cat([context, forecast], dim=1)
        flat_images = images.reshape(-1, *images.shape[2:])
        with torch.amp.autocast("cuda", enabled=self.use_amp):  # FIX 8: deprecated torch.cuda.amp.autocast
            x_recon, mu, logvar = vae(flat_images)
            loss, _, _ = vae.vae_loss(
                flat_images, x_recon, mu, logvar,
                beta=self.vae_beta, ssim_weight=self.vae_ssim_weight,
            )
        return loss

    def _stage2_step(self, context: torch.Tensor, forecast: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            loss, _ = self.model(context, forecast)
        return loss

    # ------------------------------------------------------------------ #
    # Epoch loops                                                          #
    # ------------------------------------------------------------------ #

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:  # FIX 9: epoch was not a parameter
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        self.model.train()
        total_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Training",
            disable=not self.is_primary,           # FIX 10: all ranks printed a progress bar
        )

        for context, forecast in pbar:
            context  = context.to(self.device, non_blocking=True)
            forecast = forecast.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.stage == 1:
                loss = self._stage1_step(context, forecast)
            else:
                loss = self._stage2_step(context, forecast)

            self.scaler.scale(loss).backward()

            if self.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.gradient_clip_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            if self.is_primary:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(1, len(train_loader))

    def _stage1_step_eval(self, context: torch.Tensor, forecast: torch.Tensor) -> torch.Tensor:
        vae         = self._get_vae()
        images      = torch.cat([context, forecast], dim=1)
        flat_images = images.reshape(-1, *images.shape[2:])
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            mu, logvar = vae.encode(flat_images)
            x_recon    = vae.decode(mu)
            loss, _, _ = vae.vae_loss(
                flat_images, x_recon, mu, logvar, beta=self.vae_beta,
            )
        return loss

    def validate(self, val_loader: Optional[DataLoader]) -> Optional[float]:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for context, forecast in tqdm(
                val_loader,
                desc="Validation",
                disable=not self.is_primary,       # FIX 10: same as train_epoch
            ):
                context  = context.to(self.device)
                forecast = forecast.to(self.device)

                if self.stage == 1:
                    loss = self._stage1_step_eval(context, forecast)
                else:
                    loss = self._stage2_step(context, forecast)

                total_loss += loss.item()

        return total_loss / max(1, len(val_loader))

    # ------------------------------------------------------------------ #
    # Checkpointing                                                        #
    # ------------------------------------------------------------------ #
    def _load_stage1_weights(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {path}")
    
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
    
        vae_state = {
            k.removeprefix("vae."): v
            for k, v in state_dict.items()
            if k.startswith("vae.")
        }
    
        # self.model is still plain LDT at the point this is called
        missing, unexpected = self.model.vae.load_state_dict(vae_state, strict=True)
        if missing:
            raise RuntimeError(f"Missing keys loading VAE weights: {missing}")
        if self.is_primary:
            logger.info(f"Loaded Stage 1 VAE weights from {path}")

    
    def save_checkpoint(self, epoch: int, val_loss: Optional[float] = None) -> Path:
        # For stage 2, self.model is LDT (not DDP), but denoiser inside it
        # may be DDP-wrapped — unwrap that too before saving.
        inner = self._get_inner_model()
    
        # Temporarily unwrap denoiser if needed so state_dict is portable
        denoiser_was_ddp = isinstance(inner.denoiser, DDP)
        if denoiser_was_ddp:
            inner.denoiser = inner.denoiser.module
    
        checkpoint = {
            "epoch":                epoch,
            "model_state_dict":     inner.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict":    self.scaler.state_dict(),
            "config":               self.config,
            "val_loss":             val_loss,
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)
    
        # Re-wrap so training can continue after saving
        if denoiser_was_ddp:
            inner.denoiser = DDP(inner.denoiser, device_ids=[self.local_rank])
    
        if self.is_primary:
            logger.info(f"Checkpoint saved to {path}")
        return path

    # ------------------------------------------------------------------ #
    # Top-level train entry point                                          #
    # ------------------------------------------------------------------ #

    def train(self, num_epochs: Optional[int] = None):
        num_epochs               = num_epochs or self.max_epochs
        train_loader, val_loader = self.setup_data()

        if self.mlflow_enabled:
            with mlflow.start_run(run_name=self.run_name):
                mlflow.log_params(_flatten(self.config))
                self._training_loop(train_loader, val_loader, num_epochs)
        else:
            self._training_loop(train_loader, val_loader, num_epochs)

    def _training_loop(
        self,
        train_loader: DataLoader,
        val_loader:   Optional[DataLoader],
        num_epochs:   int,
    ) -> None:
        best_val_loss = float("inf")

        es_cfg      = self.config["training"].get("early_stopping", {})
        es_enabled  = es_cfg.get("enabled", False)
        es_patience = es_cfg.get("patience", 20)
        es_counter  = 0

        for epoch in range(num_epochs):
            if self.is_primary:
                logger.info(
                    f"\nEpoch {epoch + 1}/{num_epochs}  "
                    f"(lr={self.optimizer.param_groups[0]['lr']:.2e})"
                )

            # FIX 12: all ranks must call train_epoch/validate —
            # previously only rank 0 did, so non-primary ranks hung
            # waiting for DDP gradient syncs that never came
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss   = self.validate(val_loader)

            self.scheduler.step()

            if self.is_primary:
                logger.info(f"Train Loss: {train_loss:.6f}")
                if val_loss is not None:
                    logger.info(f"Val Loss: {val_loss:.6f}")

                if self.mlflow_enabled:
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("lr", self.optimizer.param_groups[0]["lr"], step=epoch)
                    if val_loss is not None:
                        mlflow.log_metric("val_loss", val_loss, step=epoch)

                if (epoch + 1) % self.save_every == 0:
                    self.save_checkpoint(epoch + 1, val_loss)

                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    es_counter    = 0
                    self.save_checkpoint(epoch + 1, val_loss)
                    logger.info(f"New best model — Val Loss: {val_loss:.6f}")
                elif es_enabled:
                    es_counter += 1
                    logger.info(f"No improvement ({es_counter}/{es_patience})")
                    if es_counter >= es_patience:
                        logger.info("Early stopping triggered.")
                        break


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the CPDiT latent diffusion transformer")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    config  = load_config(args.config)
    trainer = Trainer(config, device=device)
    trainer.train()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
