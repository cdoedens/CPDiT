"""Training entry point for the latent diffusion transformer."""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.models import LatentDiffusionTransformer
from src.petdata import build_dataloader

from .config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BEST_CHECKPOINT_NAME = "best_model.pt"


class Trainer:
    """
    Training loop manager.

    Training is staged, not single-pass. `training.vae_loss_weight`,
    `training.diffusion_loss_weight`, `training.freeze_vae` and
    `training.detach_latents` select which stage this run is:

        stage 1  configs/stage1_vae.yaml     VAE alone
        stage 2  configs/train_config.yaml   diffusion, frozen VAE
        stage 3  configs/stage3_joint.yaml   joint fine-tune, both guards on

    The staging is not stylistic. The diffusion objective reduces to
    epsilon-MSE, so an encoder that is free to shrink `mu` towards zero can
    drive that loss to zero while destroying the latent space — see the module
    docstring in src/models/latent_diffusion.py.
    """

    def __init__(self, config: dict, device: str = "cuda"):
        self.config     = config
        self.device     = device
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank       = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_primary = self.rank == 0

        model_cfg       = config["model"]
        training_cfg    = config["training"]
        optimiser_cfg   = config["optimiser"]
        transformer_cfg = model_cfg.get("transformer", {})
        diffusion_cfg   = model_cfg.get("diffusion", {})
        data_cfg        = config["data"]

        self.gradient_clip_norm = training_cfg.get("gradient_clip_norm", 1.0)
        self.checkpoint_dir     = Path(training_cfg["checkpoint_dir"])
        self.save_every         = training_cfg.get("save_every_n_epochs", 5)
        self.keep_last_n        = training_cfg.get("keep_last_n_checkpoints", 3)
        self.vae_beta           = training_cfg.get("vae_beta", 0.01)
        self.vae_ssim_weight    = training_cfg.get("vae_ssim_weight", 0.1)
        self.vae_loss_weight    = training_cfg.get("vae_loss_weight", 0.1)
        self.accum_steps        = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
        self.irradiance_channel = data_cfg.get("irradiance_channel", 0)
        self.max_epochs         = int(training_cfg.get("max_epochs", 100))

        # ── Stage controls ────────────────────────────────────────────────
        # stage 1  vae_loss_weight > 0, diffusion_loss_weight = 0, VAE trainable
        # stage 2  vae_loss_weight = 0, diffusion_loss_weight = 1, VAE frozen
        # stage 3  both > 0, VAE unfrozen at a reduced LR, detach off, and
        #          model.latent_norm = "batch" plus a pixel_loss_weight to make
        #          the joint gradient safe. See the module docstring in
        #          latent_diffusion.py for why those two are not optional.
        self.diffusion_loss_weight = float(training_cfg.get("diffusion_loss_weight", 1.0))
        self.detach_latents        = bool(training_cfg.get("detach_latents", True))
        self.pixel_loss_weight     = float(training_cfg.get("pixel_loss_weight", 0.0))
        self.pixel_loss_max_t      = float(training_cfg.get("pixel_loss_max_t", 0.3))
        self.freeze_vae            = bool(training_cfg.get("freeze_vae", False))
        self.vae_lr_mult           = float(training_cfg.get("vae_lr_mult", 1.0))
        self.vae_max_frames        = training_cfg.get("vae_max_frames", None)

        if not self.detach_latents and model_cfg.get("latent_norm", "ema") != "batch":
            logger.warning(
                "detach_latents is off but model.latent_norm is not 'batch'. "
                "Collapsing the latent is then the global minimum of the diffusion "
                "loss, which is exactly the failure this guard exists to prevent. "
                "Set model.latent_norm: batch, or leave detach_latents on."
            )

        eval_cfg = config.get("evaluation", {})
        self.forecast_eval_batches = int(eval_cfg.get("forecast_eval_batches", 0))
        self.forecast_num_steps    = int(eval_cfg.get("num_steps", 100))
        self.forecast_sampler      = eval_cfg.get("sampler", "ode")
        self.max_val_batches       = int(eval_cfg.get("max_val_batches", 0))

        self._set_seed(training_cfg.get("seed", 42))

        if self.is_primary:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            Path(training_cfg["log_dir"]).mkdir(parents=True, exist_ok=True)

        # ── Build model ────────────────────────────────────────────────────
        self.model = LatentDiffusionTransformer(
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
            latent_norm            = model_cfg.get("latent_norm", "ema"),
        ).to(device)

        # Freeze before the optimiser is built: AdamW below filters on
        # requires_grad, so this ordering is what keeps frozen VAE parameters
        # out of the optimiser (and out of DDP's reducer) entirely.
        if self.freeze_vae:
            self.model.freeze_vae()
            if self.is_primary:
                logger.info("VAE frozen — training the diffusion stack only.")

        if self.is_primary:
            n_params = sum(p.numel() for p in self.model.parameters())
            n_train  = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(
                "Model parameters: %.1fM (%.1fM trainable)", n_params / 1e6, n_train / 1e6
            )
            logger.info("Denoiser: %s", self.model.denoiser.extra_repr())
            logger.info(
                "Stage — vae_loss_weight=%.3g diffusion_loss_weight=%.3g "
                "pixel_loss_weight=%.3g detach_latents=%s freeze_vae=%s latent_norm=%s",
                self.vae_loss_weight, self.diffusion_loss_weight,
                self.pixel_loss_weight, self.detach_latents, self.freeze_vae,
                model_cfg.get("latent_norm", "ema"),
            )

        # ── DDP ────────────────────────────────────────────────────────────
        if dist.is_initialized():
            # With vae_loss_weight == 0 the VAE decoder never participates in
            # the loss, and with diffusion_loss_weight == 0 the context encoder
            # and denoiser do not either, so DDP must be told to expect unused
            # parameters in both cases.
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=(
                    self.vae_loss_weight == 0 or self.diffusion_loss_weight == 0
                ),
            )

        # ── Optional torch.compile (applied after DDP, per PyTorch docs) ────
        # TF32 for the fp32 operations autocast leaves alone (optimiser maths,
        # norms). Free accuracy-for-speed on Ampere and newer; ignored on older
        # cards.
        if device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8:
            torch.set_float32_matmul_precision(
                training_cfg.get("matmul_precision", "high")
            )

        if training_cfg.get("compile_model", False):
            # "reduce-overhead" turns on CUDA graphs. This graph does not capture
            # cleanly — inductor partitions it around device copies and then
            # segfaults on H200 — and CUDA graphs buy little at this step time
            # anyway, so the default mode is used unless asked otherwise.
            compile_mode = training_cfg.get("compile_mode", "default")
            if self.is_primary:
                logger.info("Compiling model with torch.compile (mode=%s)...", compile_mode)
            # DDPOptimizer splits the compiled graph at DDP's gradient buckets
            # so the all-reduce for early buckets overlaps the rest of the
            # backward. That overlap is most of what makes multi-GPU scale, so
            # it stays on by default; set optimize_ddp: false only if inductor
            # chokes on the split graph.
            torch._dynamo.config.optimize_ddp = bool(
                training_cfg.get("optimize_ddp", True)
            )
            # Static shapes. `dynamic=True` forces symbolic shapes AND lifts the
            # float keyword arguments of `forward` into the graph as tensors, so
            # `diffusion_loss_weight > 0` becomes an in-graph `.item()`/`gt`
            # rather than a Python branch folded at trace time. Those scalars —
            # a bool, and T_ctx / T_total read off `.shape[1]` — then have to
            # cross DDPOptimizer's bucket split, and inductor's
            # FakifiedOutWrapper asks every output node for `.meta["val"]`:
            #   AttributeError: 'bool' object has no attribute 'meta'
            # which is what killed stage 3 the moment the VAE became trainable
            # and DDP put buckets inside the encoder. (Stage 2 escaped it only
            # because a frozen VAE leaves no bucket boundary there, and the
            # validation path escapes it by calling the unwrapped module — see
            # the comment in `validate`.) Reproduced and fixed on torch 2.10;
            # `dynamic=True` fails with optimize_ddp off as well, on a
            # symbolic-shape error inside inductor, so this is the setting to
            # change rather than the bucket split, which is worth keeping.
            #
            # Nothing here needs dynamic shapes: dataloader.drop_last is true,
            # so every batch the compiled wrapper sees has identical shape.
            # Validation and forecast_metrics run the unwrapped module. A shape
            # that does vary costs a recompile, not a failure.
            kwargs = {"dynamic": bool(training_cfg.get("compile_dynamic", False))}
            if compile_mode not in ("default", None, ""):
                kwargs["mode"] = compile_mode
            self.model = torch.compile(self.model, **kwargs)

        # ── Optimiser ─────────────────────────────────────────────────────
        opt_cfg = optimiser_cfg.get("unified", optimiser_cfg)
        betas   = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        base_lr = opt_cfg["lr"]

        # A stage-3 joint fine-tune wants the pretrained encoder to move much
        # more slowly than the freshly-trained diffusion stack, so the VAE gets
        # its own parameter group. At the default multiplier of 1.0 this is
        # exactly one group, as before.
        # Unwrapped: by this point self.model may be DDP- and/or compile-wrapped,
        # and DDP does not forward attribute access, so `self.model.vae` would
        # raise. The parameter objects are shared, so grouping on the inner
        # module still selects exactly the right tensors.
        inner    = self._get_inner_model()
        vae_ids  = {id(p) for p in inner.vae.parameters()}
        vae_par  = [p for p in inner.vae.parameters() if p.requires_grad]
        rest_par = [
            p for p in inner.parameters()
            if p.requires_grad and id(p) not in vae_ids
        ]
        groups = [{"params": rest_par, "lr": base_lr}]
        if vae_par and self.vae_lr_mult != 1.0:
            groups.append({"params": vae_par, "lr": base_lr * self.vae_lr_mult})
            if self.is_primary:
                logger.info("VAE parameter group at lr x%.3g", self.vae_lr_mult)
        elif vae_par:
            groups[0]["params"] = rest_par + vae_par

        self.optimizer = AdamW(
            groups,
            lr           = base_lr,
            weight_decay = opt_cfg.get("weight_decay", 1e-4),
            betas        = betas,
        )

        sched_cfg     = optimiser_cfg.get("scheduler", {})
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 0))
        min_lr        = sched_cfg.get("min_lr", 1e-6)
        # Early stopping must not count strikes while the LR is still ramping:
        # a run whose best score was set at 20% of the target LR has not been
        # given a chance yet.
        self.warmup_epochs = warmup_epochs

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, self.max_epochs - warmup_epochs)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # ── Precision ─────────────────────────────────────────────────────
        # bfloat16 has the dynamic range of fp32, so it needs no loss scaling;
        # a GradScaler there costs a host sync per step for nothing. Only fp16
        # gets the scaler.
        precision = str(training_cfg.get("precision", "")).lower()
        if not precision:
            precision = "bf16" if training_cfg.get("mixed_precision", False) else "fp32"
        if not device.startswith("cuda"):
            precision = "fp32"

        # bf16 needs Ampere (sm_80+) tensor cores. On older cards torch still
        # *runs* it — torch.cuda.is_bf16_supported() returns True, meaning
        # "works", not "is fast" — but it falls off the tensor-core path
        # entirely. Measured on a V100 (sm_70): fp16 92.1 TFLOP/s, fp32 13.8,
        # bf16 10.4. Selecting bf16 there is an 8.8x slowdown against fp16 and
        # is even slower than plain fp32, which is a silent and very expensive
        # mistake on the gpuvolta queue. fp16 carries the same memory saving
        # and the GradScaler below already makes it safe.
        if precision == "bf16" and device.startswith("cuda"):
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                if self.is_primary:
                    logger.warning(
                        "bf16 requested on %s (sm_%d%d), which has no bf16 tensor "
                        "cores — it would run ~9x slower than fp16 and slower than "
                        "fp32. Using fp16 instead. Set precision: fp32 to override.",
                        torch.cuda.get_device_name(), *torch.cuda.get_device_capability(),
                    )
                precision = "fp16"

        self.precision = precision
        self.use_amp   = precision in ("bf16", "fp16")
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
        self.scaler    = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
        if self.is_primary:
            logger.info("Precision: %s", precision)

        logging_cfg         = config.get("logging", {})
        self.run_name       = logging_cfg.get("project_name", "baseline")
        self.mlflow_enabled = HAS_MLFLOW and self.is_primary
        if self.mlflow_enabled:
            mlflow.set_tracking_uri(config.get("tracking_uri", "http://localhost:5000"))
            mlflow.set_experiment(logging_cfg.get("experiment_name", "CPDiT"))

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _set_seed(self, seed: int) -> None:
        """Seed per rank so ranks draw different noise but stay reproducible."""
        seed = int(seed) + self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _get_inner_model(self) -> LatentDiffusionTransformer:
        """Unwrap DDP and/or torch.compile to get the raw LDT."""
        m = self.model
        for _ in range(3):
            if hasattr(m, "_orig_mod"):
                m = m._orig_mod
            elif isinstance(m, DDP):
                m = m.module
            else:
                break
        return m

    def _autocast(self):
        if not self.use_amp:
            return contextlib.nullcontext()
        return torch.amp.autocast("cuda", dtype=self.amp_dtype)

    def _synced_batches(self, loader: DataLoader) -> Iterator:
        """
        Iterate a loader so every DDP rank stops on the same step.

        The dataset is an IterableDataset that silently drops samples with
        missing data or NaNs, so ranks do not produce identical batch counts.
        Without this guard the first rank to run dry would leave the others
        blocked forever in the next gradient all-reduce.
        """
        iterator = iter(loader)
        while True:
            try:
                batch = next(iterator)
                have  = 1
            except StopIteration:
                batch = None
                have  = 0

            if self.world_size > 1:
                flag = torch.tensor([have], device=self.device, dtype=torch.int32)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                if int(flag.item()) == 0:
                    return
            elif have == 0:
                return

            yield batch

    def _reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Average a metric dict across DDP ranks so every rank agrees."""
        if self.world_size == 1 or not metrics:
            return metrics
        keys   = sorted(metrics)
        tensor = torch.tensor(
            [metrics[k] for k in keys], device=self.device, dtype=torch.float64
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return dict(zip(keys, tensor.tolist()))

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #

    def setup_data(self) -> tuple[DataLoader, Optional[DataLoader]]:
        if self.is_primary:
            logger.info("Setting up datasets...")
        train_loader = build_dataloader("train", self.config, shuffle=True)
        # drop_last on validation throws away up to batch_size-1 samples for no
        # benefit, and on a small split can discard every batch there is.
        val_loader   = build_dataloader("val", self.config, shuffle=False,
                                        drop_last=False)
        return train_loader, val_loader

    # ------------------------------------------------------------------ #
    # Epoch loops                                                          #
    # ------------------------------------------------------------------ #

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict[str, float]:
        dataset = train_loader.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        if hasattr(getattr(train_loader, "sampler", None), "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        self.model.train()
        totals  = {
            "total": 0.0, "diffusion": 0.0, "vae": 0.0, "recon": 0.0, "kl": 0.0,
            "pixel": 0.0, "latent_scale_ratio": 0.0,
        }
        n_steps = 0

        # Split wall-clock into "waiting for a batch" and "computing on it".
        # This is the number that says whether the GPUs are being used: the
        # loss curve cannot distinguish a fed GPU from a starved one.
        t_data = 0.0
        t_compute = 0.0
        n_samples = 0
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.perf_counter()
        wait_start = time.perf_counter()

        pbar = tqdm(
            self._synced_batches(train_loader),
            desc="Training",
            disable=not self.is_primary,
        )

        self.optimizer.zero_grad(set_to_none=True)

        for step, (context, forecast) in enumerate(pbar):
            t_data += time.perf_counter() - wait_start
            compute_start = time.perf_counter()

            context  = context.to(self.device, non_blocking=True)
            forecast = forecast.to(self.device, non_blocking=True)
            n_samples += context.shape[0]

            is_accum_boundary = (step + 1) % self.accum_steps == 0

            # Skip DDP's gradient all-reduce on non-boundary micro-steps.
            sync_ctx = (
                self.model.no_sync()
                if (self.world_size > 1 and not is_accum_boundary
                    and hasattr(self.model, "no_sync"))
                else contextlib.nullcontext()
            )

            with sync_ctx:
                with self._autocast():
                    out = self.model(
                        context, forecast,
                        vae_loss_weight       = self.vae_loss_weight,
                        vae_beta              = self.vae_beta,
                        vae_ssim_weight       = self.vae_ssim_weight,
                        diffusion_loss_weight = self.diffusion_loss_weight,
                        detach_latents        = self.detach_latents,
                        vae_max_frames        = self.vae_max_frames,
                        pixel_loss_weight     = self.pixel_loss_weight,
                        pixel_loss_max_t      = self.pixel_loss_max_t,
                        irradiance_channel    = self.irradiance_channel,
                    )
                    loss = out["loss"] / self.accum_steps
                self.scaler.scale(loss).backward()

            if is_accum_boundary:
                if self.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.gradient_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            # .item() below already synchronises, so the timing is honest and
            # does not need an extra torch.cuda.synchronize().
            totals["total"]     += out["loss"].item()
            totals["diffusion"] += out["diffusion_loss"].item()
            totals["vae"]       += out["vae_loss"].item()
            totals["recon"]     += out["recon_loss"].item()
            totals["kl"]        += out["kl_loss"].item()
            totals["pixel"]     += out["pixel_loss"].item()
            totals["latent_scale_ratio"] += out["latent_scale_ratio"].item()
            n_steps += 1

            t_compute += time.perf_counter() - compute_start

            if self.is_primary:
                elapsed = max(time.perf_counter() - epoch_start, 1e-9)
                pbar.set_postfix({
                    "loss":  f"{out['loss'].item():.4f}",
                    "diff":  f"{out['diffusion_loss'].item():.4f}",
                    "smp/s": f"{n_samples / elapsed:.2f}",
                    "wait":  f"{100 * t_data / elapsed:.0f}%",
                })

            wait_start = time.perf_counter()

        # With num_workers > 0 the dataset object here is the parent's copy and
        # its counters stay at zero — the real iteration happens in the worker
        # processes, which have their own copies.
        if (hasattr(dataset, "log_epoch_summary") and self.is_primary
                and train_loader.num_workers == 0):
            dataset.log_epoch_summary()

        if n_steps == 0:
            # Zero batches used to be averaged as max(1, 0), reporting a
            # flawless 0.0 loss for an epoch that never touched any data — a
            # run could "train" for 100 epochs on nothing and look perfect.
            raise RuntimeError(
                "The training dataloader produced no batches. With an "
                "IterableDataset every worker batches independently, so a split "
                "needs at least batch_size x num_workers samples before any "
                "batch appears, and drop_last then discards each worker's "
                "remainder. Check data.splits.train, data.sample_stride "
                f"(currently {self.config['data'].get('sample_stride', 1)}), "
                f"dataloader.batch_size x num_workers "
                f"({train_loader.batch_size} x {train_loader.num_workers}), and "
                "dataloader.drop_last."
            )

        if self.is_primary and n_steps:
            wall = max(time.perf_counter() - epoch_start, 1e-9)
            peak = (
                torch.cuda.max_memory_allocated() / 2**30
                if self.device.startswith("cuda") else 0.0
            )
            logger.info(
                "Throughput — %.2f samples/s/rank (%.2f global) | %.2fs/step | "
                "data wait %.0f%% | compute %.0f%% | peak mem %.1f GiB",
                n_samples / wall, n_samples * self.world_size / wall,
                wall / n_steps, 100 * t_data / wall, 100 * t_compute / wall, peak,
            )
            if t_data > 0.5 * wall:
                logger.warning(
                    "The GPU spent %.0f%% of the epoch waiting for data. Raise "
                    "dataloader.num_workers or data.sample_stride — more GPUs "
                    "will not help until this is under control.",
                    100 * t_data / wall,
                )

        n = max(1, n_steps)
        metrics = self._reduce_metrics({k: v / n for k, v in totals.items()})
        metrics["samples_per_s"] = n_samples / max(time.perf_counter() - epoch_start, 1e-9)
        return metrics

    @torch.no_grad()
    def validate(self, val_loader: Optional[DataLoader]) -> Optional[dict[str, float]]:
        if val_loader is None:
            return None

        self.model.eval()
        # Keys are collected from whatever the model returns rather than fixed
        # up front: stage 1 skips the denoiser, so it has no irradiance metrics
        # to report and must not contribute zeros to an average.
        totals: dict[str, float] = {}
        forecast_totals: dict[str, float] = {}
        n_batches = 0
        n_forecast_batches = 0

        inner = self._get_inner_model()

        for context, forecast in tqdm(
            self._synced_batches(val_loader), desc="Validation", disable=not self.is_primary
        ):
            if self.max_val_batches and n_batches >= self.max_val_batches:
                break

            context  = context.to(self.device, non_blocking=True)
            forecast = forecast.to(self.device, non_blocking=True)

            with self._autocast():
                # One pass gives the losses and the pixel metrics: deterministic
                # timesteps and a fixed noise draw make it comparable epoch to
                # epoch. The pixel *loss* stays off here — validation only ever
                # measures.
                #
                # `inner`, not `self.model`: the wrapper is DDP + torch.compile,
                # and this call's kwargs (deterministic, return_pixel_metrics)
                # trace a different graph from the training step, so it forces a
                # fresh compile under no_grad. DDPOptimizer's bucket split feeds
                # a plain Python int into the AOT inference path there and dies
                # with "'int' object has no attribute 'meta'". Neither wrapper
                # earns its keep here: no_grad means no gradient all-reduce, and
                # the metrics are reduced explicitly below. `forecast_metrics`
                # already runs on `inner` for the same reason.
                out = inner(
                    context, forecast,
                    vae_loss_weight       = self.vae_loss_weight,
                    vae_beta              = self.vae_beta,
                    vae_ssim_weight       = self.vae_ssim_weight,
                    diffusion_loss_weight = self.diffusion_loss_weight,
                    detach_latents        = self.detach_latents,
                    vae_max_frames        = self.vae_max_frames,
                    deterministic         = True,
                    return_pixel_metrics  = True,
                    irradiance_channel    = self.irradiance_channel,
                )

            named = {
                "total_loss": out["loss"], "diff_loss": out["diffusion_loss"],
                "vae_loss":   out["vae_loss"], "recon_loss": out["recon_loss"],
                "kl_loss":    out["kl_loss"],
                "latent_scale_ratio": out["latent_scale_ratio"],
            }
            for key in ("irradiance_mse", "irradiance_mae"):
                if key in out:
                    named[key] = out[key]
            for k, v in named.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            n_batches += 1

            # Real forecast skill via the full reverse process, on a small fixed
            # subset because it is far more expensive than the single-step
            # estimate above.
            #
            # Deliberately OUTSIDE the autocast: this integrates 100+ sequential
            # solver steps, and bf16's 8 mantissa bits accumulate real error
            # over a chain that long. The metric gating early stopping should
            # not be a measure of the sampler's rounding.
            if n_forecast_batches < self.forecast_eval_batches:
                fm = inner.forecast_metrics(
                    context, forecast,
                    num_steps=self.forecast_num_steps,
                    sampler=self.forecast_sampler,
                    irradiance_channel=self.irradiance_channel,
                )
                for k, v in fm.items():
                    forecast_totals[k] = forecast_totals.get(k, 0.0) + v.item()
                n_forecast_batches += 1

        if n_batches == 0:
            # Dividing by max(1, 0) would report a flawless 0.0 for every
            # metric, and best-model selection would happily save that as the
            # best model ever seen. No data means no metric.
            logger.warning(
                "Validation produced no batches — check data.splits.val, "
                "data.sample_stride, and dataloader.drop_last. Skipping "
                "validation, best-model selection and early stopping this epoch."
            )
            return None

        metrics = {k: v / n_batches for k, v in totals.items()}
        if n_forecast_batches:
            metrics.update({k: v / n_forecast_batches for k, v in forecast_totals.items()})

        return self._reduce_metrics(metrics)

    # ------------------------------------------------------------------ #
    # Benchmarking                                                         #
    # ------------------------------------------------------------------ #

    def benchmark(
        self,
        batch_sizes: Optional[list[int]] = None,
        steps: int = 12,
        warmup: int = 4,
    ) -> list[dict]:
        """
        Measure GPU-side throughput on synthetic batches.

        The real dataloader reads ~20s of satellite archive per sample, which
        completely masks how fast the model itself is. Feeding it random tensors
        of the correct shape removes that and answers two separate questions:

          1. How many samples/s can this GPU actually push through the model?
             Compare against the loader's rate to see which side is the limit.
          2. How large a batch fits? At 143 GiB an H200 has room the config is
             not using, and larger batches amortise the per-step overheads that
             dominate at batch size 2.

        Each batch size is timed independently. Note torch.compile recompiles
        on the first step of each new shape (the trainer compiles with static
        shapes — see the `dynamic` comment in `__init__`), which is why
        `warmup` steps are discarded.
        """
        data_cfg  = self.config["data"]
        model_cfg = self.config["model"]
        T_ctx     = data_cfg["n_prior_sat"]
        T_fcast   = data_cfg["n_post"]
        C         = model_cfg["image_channels"]
        H = W     = model_cfg["image_size"]

        if batch_sizes is None:
            batch_sizes = [self.config.get("dataloader", {}).get("batch_size", 2)]

        results = []
        self.model.train()

        for bs in batch_sizes:
            try:
                context  = torch.randn(bs, T_ctx,   C, H, W, device=self.device)
                forecast = torch.randn(bs, T_fcast, C, H, W, device=self.device)

                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                for i in range(warmup + steps):
                    if i == warmup and self.device.startswith("cuda"):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()

                    with self._autocast():
                        # Same stage settings as a real step, so the measurement
                        # reflects what will actually run — the stage-3 pixel
                        # loss in particular puts the decoder in the graph and
                        # changes the memory picture substantially.
                        out = self.model(
                            context, forecast,
                            vae_loss_weight       = self.vae_loss_weight,
                            vae_beta              = self.vae_beta,
                            vae_ssim_weight       = self.vae_ssim_weight,
                            diffusion_loss_weight = self.diffusion_loss_weight,
                            detach_latents        = self.detach_latents,
                            vae_max_frames        = self.vae_max_frames,
                            pixel_loss_weight     = self.pixel_loss_weight,
                            pixel_loss_max_t      = self.pixel_loss_max_t,
                            irradiance_channel    = self.irradiance_channel,
                        )
                    self.scaler.scale(out["loss"]).backward()
                    if self.gradient_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=self.gradient_clip_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / steps
                peak = (
                    torch.cuda.max_memory_allocated() / 2**30
                    if self.device.startswith("cuda") else 0.0
                )
                results.append({
                    "batch_size":     bs,
                    "s_per_step":     dt,
                    "samples_per_s":  bs / dt,
                    "peak_mem_gib":   peak,
                })
                if self.is_primary:
                    logger.info(
                        "batch=%3d | %6.3f s/step | %6.2f samples/s/rank | peak %5.1f GiB",
                        bs, dt, bs / dt, peak,
                    )
            except torch.cuda.OutOfMemoryError:
                if self.is_primary:
                    logger.warning("batch=%d | out of memory", bs)
                torch.cuda.empty_cache()
                break
            finally:
                del context, forecast
                self.optimizer.zero_grad(set_to_none=True)

        return results

    # ------------------------------------------------------------------ #
    # Cache priming                                                        #
    # ------------------------------------------------------------------ #

    def prime_cache(self, splits: tuple[str, ...] = ("train", "val")) -> None:
        """Deprecated shim: priming needs no model, so it is a free function."""
        prime_cache(self.config, splits=splits, is_primary=self.is_primary)

    # ------------------------------------------------------------------ #
    # Checkpointing                                                        #
    # ------------------------------------------------------------------ #

    def _checkpoint_payload(self, epoch: int, val_metrics) -> dict:
        return {
            "epoch":                epoch,
            "model_state_dict":     self._get_inner_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict":    self.scaler.state_dict(),
            "config":               self.config,
            "val_metrics":          val_metrics,
        }

    def save_checkpoint(
        self,
        epoch:       int,
        val_metrics: Optional[dict[str, float]] = None,
        is_best:     bool = False,
    ) -> Path:
        """
        Write a checkpoint. The best model goes to a stable filename so it can
        actually be found later; periodic checkpoints are epoch-numbered and
        rotated.
        """
        payload = self._checkpoint_payload(epoch, val_metrics)
        name    = BEST_CHECKPOINT_NAME if is_best else f"checkpoint_epoch_{epoch:03d}.pt"
        path    = self.checkpoint_dir / name

        # Write to a temporary file first so an interrupted job cannot leave a
        # truncated checkpoint behind.
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

        logger.info("Checkpoint saved to %s", path)
        if not is_best:
            self._rotate_checkpoints()
        return path

    def _rotate_checkpoints(self) -> None:
        if not self.keep_last_n or self.keep_last_n <= 0:
            return
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        for stale in checkpoints[:-self.keep_last_n]:
            with contextlib.suppress(OSError):
                stale.unlink()
                logger.debug("Removed old checkpoint %s", stale)

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Load a checkpoint and return the epoch it was saved at."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self._get_inner_model().load_state_dict(checkpoint["model_state_dict"])
        self._reapply_fixed_latent_scale()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        epoch = checkpoint.get("epoch", 0)
        if self.is_primary:
            logger.info("Resumed from checkpoint %s (epoch %d)", path, epoch)
        return epoch

    def init_from(self, checkpoint_path: str | Path) -> None:
        """
        Load *only* the weights from a checkpoint and start a fresh run.

        This is how a stage begins from the previous stage's result. `--resume`
        is the wrong tool: it also restores the optimiser moments, the LR
        schedule and the epoch counter, all of which belong to the run that
        produced the checkpoint, not to the new stage that is about to start
        with a different objective and learning rate.

        Loaded non-strictly so a stage-1 VAE-only checkpoint can seed a stage-2
        run even if the diffusion stack was reshaped in between.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self._get_inner_model().load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        self._reapply_fixed_latent_scale()
        if self.is_primary:
            logger.info(
                "Initialised weights from %s (epoch %s); %d missing, %d unexpected keys",
                path, checkpoint.get("epoch", "?"), len(missing), len(unexpected),
            )
            if missing:
                logger.info("  missing (left at init): %s", sorted(missing)[:8])
            if unexpected:
                logger.info("  unexpected (ignored): %s", sorted(unexpected)[:8])

    def _reapply_fixed_latent_scale(self) -> None:
        """
        Restore a config-pinned `latent_scale` after a state-dict load.

        `latent_std` is a buffer, so loading a checkpoint overwrites it with
        whatever the *previous* stage's EMA had drifted to — silently discarding
        the value just measured with scripts/measure_latent_scale.py and putting
        the sampler's prior back out of step with the data.
        """
        configured = self.config["model"].get("latent_scale")
        if configured is None:
            return
        inner = self._get_inner_model()
        inner.latent_std.fill_(float(configured))
        inner.latent_scale_initialised.fill_(True)
        if self.is_primary:
            logger.info(
                "Re-applied fixed model.latent_scale=%g after load", float(configured)
            )

    # ------------------------------------------------------------------ #
    # Top-level train entry point                                          #
    # ------------------------------------------------------------------ #

    def train(
        self,
        num_epochs:   Optional[int] = None,
        resume_from:  Optional[str] = None,
        init_from:    Optional[str] = None,
    ):
        num_epochs               = num_epochs or self.max_epochs
        train_loader, val_loader = self.setup_data()

        start_epoch = 0
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)
        elif init_from is not None:
            self.init_from(init_from)

        if self.mlflow_enabled:
            with mlflow.start_run(run_name=self.run_name):
                mlflow.log_params(_flatten(self.config))
                self._training_loop(train_loader, val_loader, num_epochs, start_epoch)
        else:
            self._training_loop(train_loader, val_loader, num_epochs, start_epoch)

    def _training_loop(
        self,
        train_loader: DataLoader,
        val_loader:   Optional[DataLoader],
        num_epochs:   int,
        start_epoch:  int = 0,
    ) -> None:
        best_metric = float("inf")

        es_cfg      = self.config["training"].get("early_stopping", {})
        es_enabled  = es_cfg.get("enabled", False)
        es_patience = es_cfg.get("patience", 20)
        es_counter  = 0
        monitor     = es_cfg.get("monitor", "irradiance_mse")

        for epoch in range(start_epoch, num_epochs):
            if self.is_primary:
                logger.info(
                    "Epoch %d/%d  (lr=%.2e)",
                    epoch + 1, num_epochs, self.optimizer.param_groups[0]["lr"],
                )

            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics   = self.validate(val_loader)

            self.scheduler.step()

            # Metrics are all-reduced, so every rank computes the same
            # decisions here. Only the file writes are rank-0, which keeps the
            # control flow identical across ranks and cannot deadlock.
            primary_metric = None
            if val_metrics is not None:
                primary_metric = val_metrics.get(monitor)
                if primary_metric is None and self.is_primary:
                    # This used to fall back to total_loss, which is dominated
                    # by the diffusion loss — the one quantity that *improves*
                    # while a collapsing latent destroys forecast skill. Silently
                    # monitoring it made the failure invisible, so refuse instead.
                    logger.error(
                        "early_stopping.monitor=%r is not in the validation "
                        "metrics %s — skipping best-model selection and early "
                        "stopping this epoch. (forecast_* keys need "
                        "evaluation.forecast_eval_batches > 0.)",
                        monitor, sorted(val_metrics),
                    )

            is_best = primary_metric is not None and primary_metric < best_metric
            if is_best:
                best_metric = primary_metric
                es_counter  = 0
            elif epoch < self.warmup_epochs:
                # The LR is still ramping; a "no improvement" here says nothing
                # about the model. The previous run stopped at epoch 11 having
                # set its best score at epoch 1 on lr=2e-5, four epochs before
                # the LR ever reached its target.
                es_counter = 0
            elif es_enabled and primary_metric is not None:
                es_counter += 1

            if self.is_primary:
                logger.info(
                    "Train — total: %.6f | diffusion: %.6f | pixel: %.6f | "
                    "vae: %.6f (recon %.6f, kl %.6f)",
                    train_metrics["total"], train_metrics["diffusion"],
                    train_metrics["pixel"], train_metrics["vae"],
                    train_metrics["recon"], train_metrics["kl"],
                )
                if val_metrics is not None:
                    logger.info(
                        "Val   — total: %.6f | diffusion: %.6f | vae: %.6f "
                        "(recon %.6f) | irradiance MSE: %s | MAE: %s",
                        val_metrics["total_loss"], val_metrics["diff_loss"],
                        val_metrics["vae_loss"], val_metrics["recon_loss"],
                        _fmt(val_metrics.get("irradiance_mse")),
                        _fmt(val_metrics.get("irradiance_mae")),
                    )
                    if "forecast_rmse" in val_metrics:
                        # Persistence is the number that matters. A forecast
                        # RMSE above it means the model is worse than copying
                        # the last observed frame, whatever the losses say.
                        skill = val_metrics["forecast_rmse"] / max(
                            val_metrics.get("persistence_rmse", float("nan")), 1e-12
                        )
                        logger.info(
                            "Val   — forecast RMSE: %.6f | MAE: %.6f || "
                            "persistence: %.6f | climatology: %.6f || "
                            "vs persistence: %.2fx %s",
                            val_metrics["forecast_rmse"], val_metrics["forecast_mae"],
                            val_metrics.get("persistence_rmse", float("nan")),
                            val_metrics.get("climatology_rmse", float("nan")),
                            skill, "(BEATS IT)" if skill < 1.0 else "(WORSE)",
                        )
                # Host-side read of the latent buffer: once per epoch here, never
                # inside the compiled step.
                self._get_inner_model().check_latent_health()
                ratio = val_metrics.get("latent_scale_ratio") if val_metrics else None
                logger.info(
                    "Latent scale (EMA std): %.4g | true/assumed std ratio: %s",
                    float(self._get_inner_model().latent_std), _fmt(ratio),
                )
                # Only meaningful once something consumes the scale. In stage 1
                # there is no diffusion and no sampling, and the EMA is expected
                # to lag a fast-moving encoder, so warning there would fire every
                # epoch of every run and train the reader to ignore it.
                if (self.diffusion_loss_weight > 0 and ratio is not None
                        and not 0.5 < ratio < 2.0):
                    logger.warning(
                        "Latent scale ratio %.3f is far from 1.0 — the sampler "
                        "starts from N(0, I) but the score network is seeing "
                        "latents of a very different scale, so samples will be "
                        "poor however low the diffusion loss goes. If the VAE is "
                        "trainable here, lower model.latent_scale_momentum "
                        "(0.9 tracked to within 1%% where 0.99 lagged 15-40%%).",
                        ratio,
                    )

                if self.mlflow_enabled:
                    for k, v in train_metrics.items():
                        mlflow.log_metric(f"train_{k}", v, step=epoch)
                    mlflow.log_metric("lr", self.optimizer.param_groups[0]["lr"], step=epoch)
                    mlflow.log_metric(
                        "latent_std", float(self._get_inner_model().latent_std), step=epoch
                    )
                    if val_metrics is not None:
                        for k, v in val_metrics.items():
                            mlflow.log_metric(f"val_{k}", v, step=epoch)

                if (epoch + 1) % self.save_every == 0:
                    self.save_checkpoint(epoch + 1, val_metrics)

                if is_best:
                    self.save_checkpoint(epoch + 1, val_metrics, is_best=True)
                    logger.info("New best model — %s: %.6f", monitor, best_metric)
                elif es_enabled and primary_metric is not None:
                    logger.info("No improvement (%d/%d)", es_counter, es_patience)

            if es_enabled and es_counter >= es_patience:
                if self.is_primary:
                    logger.info("Early stopping triggered.")
                break


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float]) -> str:
    """Format an optional metric — stage 1 has no diffusion-derived metrics."""
    return "n/a" if value is None else f"{value:.6f}"


# Batch size used while priming the cache. Priming never looks at the collated
# tensors -- it walks the stream purely so the per-frame disk cache fills -- but
# a large batch still makes EVERY worker hold that many assembled samples
# (~10 MB each) before it emits anything, and DataLoader prefetches two batches
# deep. At the training batch size of 32 that is ~650 MB per worker of pure
# overhead, which is memory that could have paid for more workers. It also makes
# the progress bar tick 16x less often, which reads as "priming got slower".
PRIME_BATCH_SIZE = 2


def prime_cache(
    config:     dict,
    splits:     tuple[str, ...] = ("train", "val"),
    is_primary: bool = True,
) -> None:
    """
    Walk the dataloader once to populate the frame cache.

    Deliberately a free function, not a Trainer method: priming touches no
    model, and constructing a Trainer builds a 163M-parameter network, an AdamW
    state for it and a torch.compile wrapper before doing any I/O at all. That
    is pure waste on a CPU-only priming job, and on a memory-bound node it is
    waste that competes with the workers actually doing the work.

    Every sample costs ~16-20s of archive I/O against ~3 ms to read back from
    /scratch, so this pass is the expensive one. The cache outlives the job, so
    run it once on the `normal` queue and every GPU job afterwards starts
    compute-bound.

    Safe to re-run and safe to interrupt: entries are written atomically and a
    partly-populated cache is simply a partly-warm one.
    """
    for split in splits:
        loader = build_dataloader(split, config, shuffle=False,
                                  drop_last=False, batch_size=PRIME_BATCH_SIZE,
                                  prime_mode=True)
        dataset = loader.dataset
        cache = getattr(dataset, "cache", None)
        if cache is not None and not cache.enabled:
            logger.warning(
                "dataloader.cache_dir is not set — priming would do nothing."
            )
            return

        span = config["data"]["splits"][split]
        logger.info(
            "Priming '%s' (%s .. %s) into %s | %d workers, batch %d ...",
            split, span.get("start"), span.get("end"), cache.root,
            loader.num_workers, loader.batch_size,
        )
        # A long run of zero batches means the archive is serving nothing for
        # this date range, not that the loader is slow; the dataset warns about
        # that directly. This bar cannot show it, because it only advances when
        # a batch actually arrives.
        t0, n = time.perf_counter(), 0
        bar = tqdm(loader, desc=f"Priming {split}", disable=not is_primary,
                   unit="batch")
        for _ in bar:
            n += 1
            if is_primary and n % 20 == 0:
                elapsed = max(time.perf_counter() - t0, 1e-9)
                bar.set_postfix({
                    "samples/s": f"{n * loader.batch_size / elapsed:.1f}",
                })
        wall = time.perf_counter() - t0
        logger.info(
            "Primed '%s': %d batches (%d anchors) in %.0fs (%.2f anchors/s)",
            split, n, n * loader.batch_size, wall,
            n * loader.batch_size / max(wall, 1e-9),
        )
        if n == 0:
            logger.error(
                "Priming '%s' produced NO samples at all. The split range "
                "(%s .. %s) is almost certainly outside what the archive "
                "serves — check it before re-running.",
                split, span.get("start"), span.get("end"),
            )
        if cache is not None and loader.num_workers == 0:
            logger.info("[%s] %s", split, cache.summary())


# Above this, a --workers override gets a warning rather than silent
# acceptance. See the comment at its call site and dataloader.num_workers in
# train_config.yaml for the measurements behind the number.
HIGH_WORKER_COUNT_THRESHOLD = 20


def warn_on_high_worker_count(n: int) -> None:
    """
    Warn when a worker count is well above what this pipeline has been tested
    at, without blocking the run — some environments genuinely have the
    headroom, so this is advisory, not a refusal.

    MEASURED: 12 workers alone reached ~35 GiB resident and was still rising
    after 2 minutes of real fetching; a --workers value naively derived from a
    node's CPU count (ncpus - 2 = 46 on a 48-CPU normal-queue node) OOM-killed
    a DataLoader worker on a 188 GiB node. Each worker re-imports the full
    pyearthtools/dask/xarray stack and keeps growing while it iterates, so this
    pipeline does not scale safely with CPU count the way a lightweight
    IterableDataset would.
    """
    if n > HIGH_WORKER_COUNT_THRESHOLD:
        logger.warning(
            "%d workers is well above the range this has been tested at "
            "(measured safe default: 10). Each worker's memory use grows "
            "during iteration, not just at import, so this can exhaust a "
            "node's memory well before its CPUs are the bottleneck. Watch "
            "`free -g` on this node while priming runs, and prefer raising "
            "this in small steps over deriving it from ncpus.",
            n,
        )


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
    parser = argparse.ArgumentParser(
        description="Train the CPDiT latent diffusion transformer"
    )
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume training from "
                             "(restores optimiser, scheduler and epoch)")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Path to a checkpoint to take WEIGHTS ONLY from, "
                             "starting a fresh run at epoch 0 with a new "
                             "optimiser. This is how stage 2 starts from a "
                             "stage-1 VAE, and stage 3 from stage 2.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override the device (e.g. cpu, cuda:0)")
    parser.add_argument("--prime-cache", action="store_true",
                        help="Populate the sample cache and exit. Needs no GPU: "
                             "run it on the normal queue so the GPU job starts "
                             "with a warm cache.")
    parser.add_argument("--workers", type=int, default=None, metavar="N",
                        help="Override dataloader.num_workers for this run. "
                             "MEMORY-bound, not just CPU-bound: each worker "
                             "re-imports the full pyearthtools/dask/xarray "
                             "stack and keeps growing while it iterates -- "
                             "measured at 12 workers reaching ~35 GiB and "
                             "still rising after 2 minutes, and 46 workers "
                             "OOM-killed a 188 GiB node. Do not scale this to "
                             "the node's CPU count; raise it in small steps "
                             "while watching `free -g` on the node.")
    parser.add_argument("--benchmark", type=str, default=None, metavar="SIZES",
                        help="Benchmark GPU throughput on synthetic batches "
                             "instead of training, e.g. --benchmark 2,4,8,16. "
                             "Isolates model speed from archive I/O.")
    args = parser.parse_args()

    # Only initialise the process group when actually launched under torchrun,
    # so single-GPU and CPU debugging runs work with a plain `python -m`.
    distributed = "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    elif args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config  = load_config(args.config)
    if args.workers is not None:
        config.setdefault("dataloader", {})["num_workers"] = max(0, args.workers)
        logger.info("dataloader.num_workers overridden to %d", args.workers)
        warn_on_high_worker_count(args.workers)
    # Priming builds no model, so it must not pay for one: constructing a
    # Trainer here would allocate a 163M-parameter network plus AdamW state and
    # wrap it in torch.compile before a single frame is read, competing for
    # memory with the workers that do the actual work.
    if args.prime_cache:
        try:
            prime_cache(config)
        finally:
            if distributed:
                dist.destroy_process_group()
        return

    trainer = Trainer(config, device=device)
    try:
        if args.benchmark:
            sizes = [int(x) for x in args.benchmark.split(",") if x.strip()]
            trainer.benchmark(batch_sizes=sizes)
        else:
            trainer.train(resume_from=args.resume, init_from=args.init_from)
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
