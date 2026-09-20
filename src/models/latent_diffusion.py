"""
Latent Diffusion Transformer for satellite image forecasting.

This is the top-level assembly. It owns the sub-modules, defines the score
function, implements the training objective, and exposes sampling. Nothing here
implements a network or an SDE — those live in `networks.py` and `diffusion.py`.

Module map
----------
    vae.py         VariationalAutoencoder   pixels <-> latent maps
    networks.py    ContextEncoder           temporal encoder (learned)
                   DiTDenoiser              score network   (learned)
    diffusion.py   VPSDE / CosineVPSDE /    forward process (no parameters)
                   VESDE, pc_sampler,       reverse-time samplers
                   ode_sampler
    THIS FILE      LatentDiffusionTransformer

Forward pass, end to end
------------------------
    context frames + target frames
        |
        v  vae.encode (one pass over every frame)
    latent maps ---> scaled by latent_std
        |                    |
        | context split      | target split
        v                    v
    latent_to_token      perturb(x0, t) via the SDE  -> x_t, z, sigma
        v                    |
    ContextEncoder ----------+
        |                    |
        +--------> DiTDenoiser(x_t, t, ctx_tokens, ctx_latents) -> out
                             |
                             v  score = -out / sigma
                   denoising score matching loss against -z/sigma

Sampling reverses this: start from the SDE's prior, integrate with the reverse
SDE or the probability-flow ODE calling `score()` at each step, then decode.

Conditioning
------------
The denoiser receives the context twice, deliberately:

  - *Spatially*, by concatenating the context latent frames to the noisy latent
    along the channel axis before patch embedding. This preserves the full
    spatial structure of the recent past, which a pooled vector cannot.
  - *Globally*, through adaLN-Zero, from the pooled temporal-transformer summary
    together with the diffusion timestep and the forecast lead time.

The lead-time embedding is what allows different forecast frames to be
distinguished; without it every frame of the horizon would be an identical
draw from the same conditional distribution.

Latent scaling
--------------
Diffusion assumes the data it corrupts has roughly unit variance, but VAE
latents have an arbitrary scale. We track an EMA of the latent standard
deviation and divide by it, which is the running-statistics analogue of Stable
Diffusion's fixed 0.18215 factor. Set ``latent_scale`` explicitly in the config
to freeze it (appropriate once the VAE has stopped moving).

Why the diffusion loss must not reach the encoder
-------------------------------------------------
``detach_latents`` defaults to True, and that default is load-bearing. The
objective below reduces to epsilon-MSE, so if the encoder is free to shrink
``mu`` towards zero then ``x_t = alpha*0 + sigma*z = sigma*z`` and the denoiser
recovers ``z = x_t/sigma`` exactly: **collapsing the latent is the global
minimum of the diffusion loss**. Nothing opposes it — reconstruction is
scale-free (the decoder simply grows its weights) and the KL pushes the same
way. A joint run without the detach did exactly this: the latent std fell 10x
over 11 epochs while the diffusion loss fell 3x and forecast skill got 2.4x
worse, because the sampler starts from N(0, I) and the score network had only
ever seen latents with a fraction of that variance.

Two mechanisms make it safe to re-enable the joint gradient (see
``latent_norm`` and ``pixel_loss_weight``):

  - ``latent_norm="batch"`` divides by a standard deviation computed *with*
    gradient attached, which makes the loss exactly scale-invariant, so
    shrinking the latent buys nothing at all.
  - ``pixel_loss_weight`` adds a decoded-x0-against-truth term at low t. A
    collapsed latent decodes to the climatological mean and scores terribly on
    it, so this is a task-aware signal that cannot be gamed by shrinking.

Score-based formulation
-----------------------
This is a score-based generative model in the SDE framework of Song et al.
(2021), not a discrete DDPM. The model learns the *score* of the perturbed data
distribution, ``s_theta(x, t) ~= grad_x log p_t(x)``, over continuous time.

  - Forward process : dx = f(x, t) dt + g(t) dw, with a Gaussian kernel
                      p(x_t | x_0) = N(alpha(t) x_0, sigma(t)^2 I)
  - Training        : denoising score matching. The kernel score is known in
                      closed form, ``-z / sigma``, so the objective is
                      ``E_t lambda(t) || s_theta(x_t, t) + z/sigma ||^2``.
  - Sampling        : reverse-time SDE with a Langevin corrector
                      (predictor-corrector), or the probability-flow ODE.

Score parameterisation
----------------------
The network emits ``out``, and the score is defined as ``s_theta = -out/sigma(t)``.
This is the standard VP parameterisation: it keeps the regression target unit
variance at every noise level, so a single network output scale works across
the whole of t, and it keeps the score finite as sigma -> 0 is approached rather
than having the network learn an O(1/sigma) blow-up. `score()` is the primary
interface; the raw output is an implementation detail.

Note this makes the default ``sigma^2``-weighted objective numerically identical
to epsilon-prediction MSE — that equivalence is a known property of the VP
family, not an accident, and it is why the switch does not destabilise training.
What is genuinely different is everything built on the score: continuous time,
the choice of SDE, the reverse-SDE/Langevin sampler, the probability-flow ODE,
and likelihood weighting (``lambda(t) = g(t)^2``), which is *not* equivalent to
epsilon-MSE and targets the exact log-likelihood bound.

References:
  - Song et al. 2021         - https://arxiv.org/abs/2011.13456 (score SDEs)
  - Song & Ermon 2019        - https://arxiv.org/abs/1907.05600 (NCSN)
  - Vincent 2011             - denoising score matching
  - Ho et al. 2020           - https://arxiv.org/abs/2006.11239 (DDPM)
  - Nichol & Dhariwal 2021   - https://arxiv.org/abs/2102.09672 (cosine)
  - Rombach et al. 2022      - https://arxiv.org/abs/2112.10752 (latent)
  - Peebles & Xie 2023       - https://arxiv.org/abs/2212.09748 (DiT)
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import SDE, broadcast_to, build_sde, ode_sampler, pc_sampler
from .networks import ContextEncoder, DiTDenoiser
from .vae import VariationalAutoencoder

logger = logging.getLogger(__name__)

# Floor on the latent standard deviation used for scaling. Reaching it means the
# latent has collapsed; scaling silently stops working past this point, so the
# model warns rather than carrying on with a broken normalisation.
LATENT_STD_FLOOR = 1e-3


class LatentDiffusionTransformer(nn.Module):
    """Latent Diffusion Transformer for probabilistic satellite image forecasting."""

    def __init__(
        self,
        image_channels:         int   = 2,
        image_size:             int   = 256,
        latent_channels:        int   = 4,
        vae_hidden_dim:         int   = 128,
        context_length:         int   = 12,
        max_forecast_steps:     int   = 32,
        # temporal transformer
        num_transformer_layers: int   = 4,
        num_heads:              int   = 8,
        feedforward_dim:        int   = 1024,
        transformer_dim:        int   = 512,
        max_seq_len:            int   = 64,
        dropout:                float = 0.1,
        # DiT denoiser
        patch_size:             int   = 4,
        denoiser_embed_dim:     int   = 768,
        denoiser_depth:         int   = 16,
        denoiser_heads:         int   = 12,
        window_size:            int   = 31,
        ffn_mult:               float = 4.0,
        cond_dim:               int   = 256,
        use_natten:             Optional[bool] = None,
        # score-based diffusion (continuous-time SDE)
        sde:                    str   = "vp_cosine",
        beta_min:               float = 0.1,
        beta_max:               float = 20.0,
        cosine_s:               float = 0.008,
        sigma_min:              float = 0.01,
        sigma_max:              float = 50.0,
        t_eps:                  Optional[float] = None,
        loss_weighting:         str   = "sigma2",
        time_scale:             float = 1000.0,
        latent_scale:           Optional[float] = None,
        latent_scale_momentum:  float = 0.99,
        latent_norm:            str   = "ema",
    ):
        super().__init__()

        # ---- 1. Construction: sub-modules, the SDE, and the latent scale ----
        self.image_channels      = image_channels
        self.latent_channels     = latent_channels
        self.latent_size         = image_size // 8
        self.context_length      = context_length
        self.max_forecast_steps  = max_forecast_steps
        self.transformer_dim     = transformer_dim
        self.latent_scale_momentum = latent_scale_momentum

        if loss_weighting not in ("sigma2", "likelihood"):
            raise ValueError(
                f"loss_weighting must be 'sigma2' or 'likelihood', got {loss_weighting!r}"
            )
        self.loss_weighting = loss_weighting

        # "ema"   divide by the running buffer — a constant w.r.t. autograd.
        # "batch" divide by this batch's own std, computed *with* gradient, so
        #         the loss is exactly invariant to the latent scale. Only that
        #         second mode is safe to combine with detach_latents=False.
        if latent_norm not in ("ema", "batch"):
            raise ValueError(
                f"latent_norm must be 'ema' or 'batch', got {latent_norm!r}"
            )
        self.latent_norm = latent_norm
        self._floor_warned = False

        # Continuous-time forward SDE. Holds only Python floats, so it needs no
        # buffers and is device-agnostic.
        self.sde: SDE = build_sde(
            sde, beta_min=beta_min, beta_max=beta_max, s=cosine_s,
            sigma_min=sigma_min, sigma_max=sigma_max, t_eps=t_eps,
        )
        self.sde_name = sde

        self._latent_flat_dim = latent_channels * self.latent_size * self.latent_size

        # ------------------------------------------------------------------ #
        # Sub-modules                                                         #
        # ------------------------------------------------------------------ #
        # 1. Pixel <-> latent compression.
        self.vae = VariationalAutoencoder(
            image_channels  = image_channels,
            latent_channels = latent_channels,
            hidden_dim      = vae_hidden_dim,
            image_size      = image_size,
        )

        # Flattens each (C, Hl, Wl) latent map into one token for the encoder.
        self.latent_to_token = nn.Linear(self._latent_flat_dim, transformer_dim)

        # 2. Temporal encoder over the context sequence.
        self.context_encoder = ContextEncoder(
            latent_dim      = transformer_dim,
            num_layers      = num_transformer_layers,
            num_heads       = num_heads,
            feedforward_dim = feedforward_dim,
            dropout         = dropout,
            max_seq_len     = max_seq_len,
        )

        # 3. The score network the sampler calls at every solver step.
        self.denoiser = DiTDenoiser(
            latent_channels    = latent_channels,
            latent_size        = self.latent_size,
            context_dim        = transformer_dim,
            num_context_frames = context_length,
            patch_size         = patch_size,
            embed_dim          = denoiser_embed_dim,
            depth              = denoiser_depth,
            num_heads          = denoiser_heads,
            window_size        = window_size,
            mlp_ratio          = ffn_mult,
            cond_dim           = cond_dim,
            max_forecast_steps = max_forecast_steps,
            use_natten         = use_natten,
            time_scale         = time_scale,
        )

        # ------------------------------------------------------------------ #
        # Latent scaling                                                      #
        # ------------------------------------------------------------------ #
        # `latent_std` is the running estimate the diffusion process divides by.
        # A user-supplied value freezes it; otherwise it is tracked by EMA
        # during training. It is a buffer so it lands in checkpoints and is
        # broadcast across DDP ranks.
        if latent_scale is not None and float(latent_scale) <= LATENT_STD_FLOOR:
            raise ValueError(
                f"latent_scale={latent_scale} is at or below the {LATENT_STD_FLOOR:.0e} "
                "scaling floor, so it would be clamped and the diffusion process "
                "would see mis-scaled data. A value this small almost always means "
                "it was measured from an already-collapsed VAE."
            )
        self.latent_scale_fixed = latent_scale is not None
        self.register_buffer(
            "latent_std", torch.tensor(float(latent_scale) if latent_scale else 1.0)
        )
        self.register_buffer("latent_scale_initialised", torch.tensor(latent_scale is not None))

    # ==========================================================================
    # 2. Latent-space plumbing
    #
    # Encode frames to latents and back, applying the latent_std rescaling so the
    # diffusion process always sees roughly unit-variance data. Also the freeze /
    # unfreeze controls for the VAE half of the joint objective.
    # ==========================================================================

    def freeze_vae(self) -> None:
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

    def unfreeze_vae(self) -> None:
        for p in self.vae.parameters():
            p.requires_grad = True
        self.vae.train()

    def _update_latent_scale(self, latents: torch.Tensor) -> None:
        """Track an EMA of the latent standard deviation (training only)."""
        if self.latent_scale_fixed or not self.training:
            return
        with torch.no_grad():
            std = latents.detach().float().std()
            if not torch.isfinite(std) or std <= 0:
                return
            if not bool(self.latent_scale_initialised):
                # Seed from the first batch so we do not spend the early epochs
                # crawling towards the right order of magnitude.
                self.latent_std.fill_(std)
                self.latent_scale_initialised.fill_(True)
            else:
                m = self.latent_scale_momentum
                self.latent_std.mul_(m).add_(std * (1.0 - m))

    def _effective_latent_std(self, dtype: torch.dtype) -> torch.Tensor:
        """
        The latent standard deviation used for scaling, floored.

        Deliberately free of any host-side inspection of the buffer: this sits
        on the per-step training path and inside the sampler's decode, and a
        `bool(tensor)` here would force a device sync every call and break the
        compiled graph. The collapse warning lives in `_update_latent_scale`,
        which already synchronises and runs once per training step.
        """
        return self.latent_std.clamp(min=LATENT_STD_FLOOR).to(dtype)

    def check_latent_health(self) -> None:
        """
        Warn, once, when the latent std has reached the scaling floor.

        Past the floor the divisor stops tracking the data, so the sampler's
        N(0, I) prior no longer matches anything the score network was trained
        on. That used to happen silently, which turned a diagnosable failure
        into a mystery.

        **Call this once per epoch, not per step.** Reading the buffer host-side
        is a `Tensor.item()`, which dynamo reports as a graph break inside the
        compiled forward — measured on an H200, so this is not hypothetical.
        The trainer calls it from the epoch summary, where it is free.
        """
        std = self.latent_std
        if self._floor_warned or float(std) > LATENT_STD_FLOOR:
            return
        self._floor_warned = True
        logger.warning(
            "Latent std %.2e has reached the %.0e floor — the latent space has "
            "collapsed and the diffusion prior no longer matches the data. "
            "Check that detach_latents is on (or latent_norm='batch'), and see "
            "the 'Why the diffusion loss must not reach the encoder' note in "
            "this module.",
            float(std), LATENT_STD_FLOOR,
        )

    def scale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents / self._effective_latent_std(latents.dtype)

    def unscale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self._effective_latent_std(latents.dtype)

    def encode_images(
        self,
        images:        torch.Tensor,   # (B, T, C, H, W)
        deterministic: bool = True,
        scaled:        bool = True,
    ) -> torch.Tensor:
        """Encode a frame sequence to spatial latent maps: (B, T, LC, Hl, Wl)."""
        assert images.ndim == 5, f"Expected (B, T, C, H, W), got {images.shape}"
        B, T = images.shape[:2]
        flat = images.reshape(B * T, *images.shape[2:])

        mu, logvar = self.vae.encode(flat)
        z = mu if deterministic else self.vae.reparameterize(mu, logvar)

        z = z.view(B, T, self.latent_channels, *z.shape[-2:])
        return self.scale_latents(z) if scaled else z

    def decode_latents(self, latents: torch.Tensor, scaled: bool = True) -> torch.Tensor:
        """Decode spatial latent maps to image frames: (B, T, C, H, W)."""
        assert latents.ndim == 5, f"Expected (B, T, LC, Hl, Wl), got {latents.shape}"
        if scaled:
            latents = self.unscale_latents(latents)
        B, T    = latents.shape[:2]
        flat    = latents.reshape(B * T, *latents.shape[2:])
        decoded = self.vae.decode(flat)
        return decoded.view(B, T, *decoded.shape[1:])

    def _encode_context(self, context_latents: torch.Tensor) -> torch.Tensor:
        """
        Project spatial latent maps to transformer tokens and encode temporally.

        Returns:
            (B, T_ctx, transformer_dim)
        """
        B, T = context_latents.shape[:2]
        tokens = self.latent_to_token(context_latents.reshape(B, T, -1))
        # No diffusion time is passed: the context is always clean, so there is
        # no noise level to condition on. Only the forecast latents are noisy,
        # and those are handled inside the denoiser.
        return self.context_encoder(tokens)

    # ==========================================================================
    # 3. The score function — the heart of the model
    #
    # Everything the sampler and the training objective need. `score()` is the
    # public interface; `score_fn()` binds the conditioning so diffusion.py can
    # call it as a plain s(x, t); `denoise_to_x0` is Tweedie's formula.
    # ==========================================================================

    def perturb(
        self,
        x:     torch.Tensor,   # (B, T, LC, Hl, Wl)
        t:     torch.Tensor,   # (B,) continuous time in [t_eps, T]
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample x_t ~ p(x_t | x_0). Returns (x_t, noise, sigma)."""
        return self.sde.perturb(x, t, noise)

    def score(
        self,
        x:               torch.Tensor,  # (B, T_fcast, LC, Hl, Wl) perturbed latents
        t:               torch.Tensor,  # (B,) continuous time
        context_tokens:  torch.Tensor,
        context_latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate ``grad_x log p_t(x)``.

        The network emits an epsilon-scaled residual and the score is
        ``-out / sigma(t)``; see the module docstring for why the model is
        parameterised this way rather than regressing the score directly.
        """
        out = self.denoiser(x, t, context_tokens, context_latents)
        _, sigma = self.sde.alpha_sigma(t)
        return -out / broadcast_to(sigma, x).clamp(min=1e-8)

    def score_fn(
        self, context_tokens: torch.Tensor, context_latents: torch.Tensor
    ):
        """Bind the conditioning, yielding the ``(x, t) -> score`` closure the
        samplers expect."""
        def fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.score(x, t, context_tokens, context_latents)
        return fn

    def denoise_to_x0(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
        clamp: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Tweedie's formula: E[x_0 | x_t] = (x_t + sigma^2 * score) / alpha.

        The score-based counterpart of DDPM eq. 15, and the reason a score model
        needs no separate x0-prediction head.

        The 1/alpha factor is inherently ill-conditioned as t -> T: for the
        cosine VP SDE alpha(1) ~ 1e-5, so any error in the score is amplified
        a hundred-thousand-fold — which is unavoidable, since x_T carries no
        information about x_0. Pass ``clamp`` to bound the estimate (latents are
        unit-scaled, so a few standard deviations is far outside the data) when
        the result feeds a metric that must stay finite and comparable.
        """
        alpha, sigma = self.sde.alpha_sigma(t)
        x0 = (
            x + broadcast_to(sigma, x) ** 2 * score
        ) / broadcast_to(alpha, x).clamp(min=1e-8)
        return x0 if clamp is None else x0.clamp(-clamp, clamp)

    def sample_times(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Draw training times ~ U(t_eps, T).

        Stratified over the batch rather than i.i.d.: with the small batch sizes
        used here, i.i.d. draws leave large parts of the time axis unvisited in
        any given step, which shows up as a noisy loss.
        """
        u = torch.rand(batch_size, device=device)
        strata = (torch.arange(batch_size, device=device) + u) / batch_size
        return self.sde.t_eps + strata * (self.sde.T - self.sde.t_eps)

    # ==========================================================================
    # 4. Training: denoising score matching
    #
    # One forward computes both objectives (diffusion + VAE) so DDP traces a single
    # graph and the batch passes through the VAE encoder exactly once.
    # ==========================================================================

    def forward(
        self,
        context_images:  torch.Tensor,   # (B, T_ctx,   C, H, W)
        target_images:   torch.Tensor,   # (B, T_fcast, C, H, W)
        vae_loss_weight: float = 0.1,
        vae_beta:        float = 0.01,
        vae_ssim_weight: float = 0.1,
        diffusion_loss_weight: float = 1.0,
        detach_latents:  bool  = True,
        vae_max_frames:  Optional[int] = None,
        pixel_loss_weight:  float = 0.0,
        pixel_loss_max_t:   float = 0.3,
        times:           Optional[torch.Tensor] = None,
        deterministic:   bool  = False,
        return_pixel_metrics: bool = False,
        irradiance_channel:   int  = 0,
    ) -> dict[str, torch.Tensor]:
        """
        One step covering both the diffusion and VAE objectives.

        Both losses are computed inside this single forward so that DDP traces
        one autograd graph — calling the VAE separately on the unwrapped module
        would leave its gradients outside DDP's reducer. The whole batch passes
        through the VAE encoder exactly once.

        Args:
            diffusion_loss_weight: 0 skips the context encoder and denoiser
                entirely, which is what makes stage-1 VAE-only training cheap.
            vae_max_frames: score the VAE loss on at most this many frames from
                each sample instead of all T_ctx + T_fcast of them. The VAE is a
                per-frame model and a sample's frames are 10 minutes apart, so
                they are near-duplicates: reconstructing all 13 costs 13x the
                decoder memory for very little extra signal. MEASURED on an
                H200 at 256px: all-frames needed 113 GiB at batch 16 and went
                OOM at 32, which forces a batch size far below what the rest of
                training uses.
            detach_latents: cut the diffusion gradient at the encoder. Leave
                this on unless ``latent_norm='batch'`` is also set — see the
                module docstring for why collapsing the latent is otherwise the
                global minimum of this objective.
            pixel_loss_weight: weight on a decoded-x0-against-truth term. Unlike
                the latent epsilon-MSE, a collapsed latent scores *badly* on
                this, so it is a task-aware signal the encoder cannot game.
            pixel_loss_max_t: only samples with t below this contribute to that
                term. Above it the 1/alpha factor in Tweedie amplifies score
                error by orders of magnitude and the gradient is noise.
            times: continuous diffusion times in [t_eps, T]. Drawn stratified
                at random when omitted.
            deterministic: use evenly-spaced times and a fixed noise draw
                instead of random ones. Validation must be reproducible across
                epochs, otherwise the metric driving best-model selection and
                early stopping is mostly noise.
            return_pixel_metrics: additionally decode the Tweedie x0 estimate
                and score it against the target in pixel space.

        Returns a dict with: loss, diffusion_loss, vae_loss, recon_loss,
        kl_loss, latent_scale_ratio, and — when the diffusion branch runs and
        pixel metrics are requested — irradiance_mse / irradiance_mae.
        """
        B      = context_images.shape[0]
        device = context_images.device
        T_ctx  = context_images.shape[1]

        # ---- One encoder pass over every frame ----------------------------
        all_images = torch.cat([context_images, target_images], dim=1)
        T_total    = all_images.shape[1]
        flat       = all_images.reshape(B * T_total, *all_images.shape[2:])

        # When the diffusion branch is off (stage 1) nothing downstream needs a
        # latent for every frame, so the subsample happens before the encoder
        # and saves both halves of the autoencoder. Otherwise every frame must
        # be encoded and only the decoder is spared.
        needs_all_latents = diffusion_loss_weight > 0
        vae_idx = self._vae_frame_indices(
            B, T_total, vae_max_frames, device, deterministic
        )

        if vae_idx is not None and not needs_all_latents:
            flat = flat[vae_idx]
            vae_idx = None                       # already applied

        mu, logvar = self.vae.encode(flat)

        # ---- VAE objective (reconstruction from a sampled z) --------------
        zero = torch.zeros((), device=device, dtype=mu.dtype)
        if vae_loss_weight > 0:
            target   = flat if vae_idx is None else flat[vae_idx]
            mu_v     = mu if vae_idx is None else mu[vae_idx]
            logvar_v = logvar if vae_idx is None else logvar[vae_idx]
            z_sample = self.vae.reparameterize(mu_v, logvar_v)
            x_recon  = self.vae.decode(z_sample)
            vae_loss, recon_loss, kl_loss = self.vae.vae_loss(
                target, x_recon, mu_v, logvar_v,
                beta=vae_beta, ssim_weight=vae_ssim_weight,
            )
        else:
            vae_loss = recon_loss = kl_loss = zero

        if not needs_all_latents:
            # Stage 1: no diffusion, so report VAE numbers and stop. The latent
            # scale still tracks, because stage 2 reads it as a starting point.
            latents_raw = mu.detach()
            self._update_latent_scale(latents_raw)
            return {
                "loss":       vae_loss_weight * vae_loss,
                "vae_loss":   vae_loss.detach(),
                "recon_loss": recon_loss.detach(),
                "kl_loss":    kl_loss.detach(),
                "diffusion_loss": zero,
                "pixel_loss":     zero,
                "latent_scale_ratio": (
                    latents_raw.float().std()
                    / self.latent_std.float().clamp(min=1e-12)
                ).detach(),
            }

        # ---- Latents ------------------------------------------------------
        # The detach is what stops the diffusion loss from reshaping the latent
        # space; see the module docstring. It is on by default and only turned
        # off in a stage-3 joint fine-tune, alongside latent_norm='batch'.
        latents_raw = (mu.detach() if detach_latents else mu).view(
            B, T_total, self.latent_channels, *mu.shape[-2:]
        )
        self._update_latent_scale(latents_raw)

        if self.latent_norm == "batch" and self.training:
            # Differentiable denominator: the loss becomes exactly invariant to
            # the latent scale, so shrinking mu cannot reduce it.
            latent_scale_value = (
                latents_raw.float().std().clamp(min=LATENT_STD_FLOOR).to(latents_raw.dtype)
            )
        else:
            # Evaluation always uses the buffer, so validation matches sampling.
            latent_scale_value = self._effective_latent_std(latents_raw.dtype)

        latents = latents_raw / latent_scale_value

        out: dict[str, torch.Tensor] = {
            "vae_loss":   vae_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "kl_loss":    kl_loss.detach(),
            # Health check: the ratio of the true latent std to the one the
            # sampler's N(0, I) prior assumes. Anything far from 1.0 means the
            # reverse process starts somewhere the score network has never been.
            "latent_scale_ratio": (
                latents_raw.detach().float().std() / self.latent_std.float().clamp(min=1e-12)
            ).detach(),
        }

        context_latents = latents[:, :T_ctx]
        target_latents  = latents[:, T_ctx:]

        encoded_context = self._encode_context(context_latents)

        noise = None
        if times is None:
            if deterministic:
                times = self._eval_times(B, device)
                generator = torch.Generator(device=device).manual_seed(0)
                noise = torch.randn(
                    target_latents.shape, device=device,
                    dtype=target_latents.dtype, generator=generator,
                )
            else:
                times = self.sample_times(B, device)

        # ---- Denoising score matching (Vincent 2011; Song et al. 2021) ----
        # The perturbation kernel's score is known exactly, -z/sigma, so the
        # intractable score-matching objective reduces to a regression onto it.
        x_t, noise, sigma = self.perturb(target_latents, times, noise=noise)
        eps_pred = self.denoiser(x_t, times, encoded_context, context_latents)
        score = -eps_pred / sigma.clamp(min=1e-8)

        # Residual of  s_theta(x_t, t) - grad log p(x_t|x_0)  scaled by sigma,
        # i.e. sigma * s_theta + z. Working in this scaled form keeps the
        # target unit-variance at every t instead of blowing up as sigma -> 0.
        residual = sigma * score + noise

        if self.loss_weighting == "sigma2":
            # lambda(t) = sigma^2: the standard variance-reduced weighting.
            per_sample = residual.flatten(1).pow(2).mean(dim=1)
        else:
            # lambda(t) = g(t)^2: likelihood weighting, which makes the loss an
            # upper bound on the negative log-likelihood rather than a
            # perceptually-tuned surrogate.
            _, g = self.sde.sde(x_t, times)
            weight = (g ** 2 / self.sde.alpha_sigma(times)[1].clamp(min=1e-8) ** 2)
            per_sample = weight * residual.flatten(1).pow(2).mean(dim=1)

        diffusion_loss = per_sample.mean()
        total = diffusion_loss_weight * diffusion_loss + vae_loss_weight * vae_loss

        out["diffusion_loss"] = diffusion_loss.detach()

        # ---- Decoded-x0 term: pixel loss (trainable) and/or metrics -------
        want_loss    = pixel_loss_weight > 0
        pixel_loss   = zero
        if want_loss or return_pixel_metrics:
            c = irradiance_channel
            target_irr = target_images[:, :, c:c + 1]

            # nullcontext, not enable_grad: the loss path must inherit the
            # ambient grad mode so `forward_eval`'s no_grad still holds.
            with contextlib.nullcontext() if want_loss else torch.no_grad():
                # Clean-latent estimate straight from the score (Tweedie). The
                # metric path bounds it, because unbounded Tweedie at large t
                # would swamp the number; the loss path does not, because the
                # clamp would zero the gradient exactly where it binds.
                pred_x0 = self.denoise_to_x0(
                    x_t, times, score, clamp=None if want_loss else 4.0
                )
                # Unscale with the same factor used above, which under
                # latent_norm='batch' is *not* the buffer.
                pred_images = self.decode_latents(
                    pred_x0 * latent_scale_value, scaled=False
                )
                pred_irr = pred_images[:, :, c:c + 1]

            if return_pixel_metrics:
                out["irradiance_mse"] = F.mse_loss(pred_irr, target_irr).detach()
                out["irradiance_mae"] = F.l1_loss(pred_irr, target_irr).detach()

            if want_loss:
                # Restrict to low t, where Tweedie is well conditioned. The mask
                # is applied as a weighted mean rather than by indexing so the
                # shapes stay static and torch.compile does not have to
                # specialise on a data-dependent size.
                mask = (times <= pixel_loss_max_t).to(pred_irr.dtype)
                denom = mask.sum().clamp(min=1.0)
                per_sample_pixel = (
                    (pred_irr - target_irr).pow(2).flatten(1).mean(dim=1)
                )
                pixel_loss = (per_sample_pixel * mask).sum() / denom
                total = total + pixel_loss_weight * pixel_loss

        out["loss"] = total
        out["pixel_loss"] = pixel_loss.detach()
        return out

    # ==========================================================================
    # 5. Evaluation
    #
    # Deterministic by construction: fixed times and a fixed noise draw, so the
    # metric driving best-model selection and early stopping is not a random
    # variable. `forecast_metrics` runs the real reverse process for true skill.
    # ==========================================================================

    def _vae_frame_indices(
        self,
        batch:         int,
        frames:        int,
        max_frames:    Optional[int],
        device:        torch.device,
        deterministic: bool,
    ) -> Optional[torch.Tensor]:
        """
        Which of each sample's frames the VAE loss should score.

        Returns flat indices into the (batch * frames) stack, or None to keep
        every frame. Frames are drawn independently per sample so a step still
        sees a spread of times of day rather than the same slot in every sample.

        Validation passes ``deterministic`` and gets an evenly spaced subset, so
        the reconstruction metric driving stage-1 early stopping is comparable
        from epoch to epoch rather than a fresh random draw each time.
        """
        if not max_frames or max_frames >= frames:
            return None

        if deterministic:
            picks = torch.linspace(0, frames - 1, max_frames, device=device).long()
            picks = picks.unsqueeze(0).expand(batch, -1)
        else:
            picks = torch.rand(batch, frames, device=device).argsort(dim=1)[:, :max_frames]

        offsets = torch.arange(batch, device=device).unsqueeze(1) * frames
        return (picks + offsets).reshape(-1)

    def _eval_times(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Deterministic times spread evenly over [t_eps, T].

        Validation must be comparable across epochs. Drawing fresh random times
        each batch makes the metric a random variable and turns best-model
        selection and early stopping into noise.
        """
        # Stratum midpoints rather than linspace endpoints. `linspace` would put
        # a sample at exactly t = T, where alpha ~ 1e-5 makes the Tweedie x0
        # estimate meaningless — and at the batch size used here that is a large
        # fraction of the evaluation batch driving early stopping.
        i = torch.arange(batch_size, device=device, dtype=torch.float32)
        return self.sde.t_eps + ((i + 0.5) / batch_size) * (self.sde.T - self.sde.t_eps)

    @torch.no_grad()
    def forward_eval(
        self,
        context_images: torch.Tensor,
        target_images:  torch.Tensor,
        irradiance_channel: int = 0,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Deterministic single-step evaluation.

        Thin wrapper over `forward` with fixed timesteps, a fixed noise draw
        and pixel metrics enabled. Kept as a separate entry point for scripts
        that only want evaluation numbers.
        """
        out = self.forward(
            context_images, target_images,
            deterministic=True, return_pixel_metrics=True,
            irradiance_channel=irradiance_channel, **kwargs,
        )
        out["latent_loss"] = out["diffusion_loss"]
        return out

    @torch.no_grad()
    def forecast_metrics(
        self,
        context_images: torch.Tensor,
        target_images:  torch.Tensor,
        num_steps: int = 100,
        irradiance_channel: int = 0,
        sampler: str = "ode",
    ) -> dict[str, torch.Tensor]:
        """
        True forecast skill: integrate the reverse process fully and score the
        result against the target in pixel space.

        Defaults to the probability-flow ODE because it is deterministic, so the
        metric reflects the model rather than the sampler's noise draw.

        Far more expensive than `forward_eval`, so the trainer runs it on a
        small fixed subset of the validation set.

        Two reference forecasts are returned alongside the model's, because an
        RMSE with nothing to compare it against says nothing about skill:

          persistence  copy the last context frame across the horizon. For a
                       10-minute nowcast this is the number to beat, and a model
                       that does not beat it has no value regardless of how its
                       training loss looks.
          climatology  predict the dataset mean. The fields are z-scored, so
                       that is exactly zero, and its RMSE is ~1 by construction.
        """
        forecast = self.sample(
            context_images,
            num_forecast_steps=target_images.shape[1],
            num_samples=1,
            num_steps=num_steps,
            sampler=sampler,
            generator=torch.Generator(device=context_images.device).manual_seed(0),
        )
        c = irradiance_channel
        pred_irr   = forecast[:,      :, c:c + 1]
        target_irr = target_images[:, :, c:c + 1]

        # Last observed frame, held constant over the whole horizon.
        persistence = context_images[:, -1:, c:c + 1].expand_as(target_irr)

        return {
            "forecast_mse":     F.mse_loss(pred_irr, target_irr),
            "forecast_mae":     F.l1_loss(pred_irr, target_irr),
            "forecast_rmse":    torch.sqrt(F.mse_loss(pred_irr, target_irr)),
            "persistence_rmse": torch.sqrt(F.mse_loss(persistence, target_irr)),
            "climatology_rmse": torch.sqrt(target_irr.pow(2).mean()),
        }

    # ==========================================================================
    # 6. Sampling
    #
    # Hands the bound score function to a sampler in diffusion.py, then decodes the
    # resulting latents back to pixels.
    # ==========================================================================

    @torch.no_grad()
    def sample(
        self,
        context_images:     torch.Tensor,   # (B, T_ctx, C, H, W)
        num_forecast_steps: int,
        num_samples:        int   = 1,
        sampler:            str   = "pc",
        num_steps:          int   = 100,
        corrector_steps:    int   = 1,
        snr:                float = 0.16,
        ode_method:         str   = "heun",
        clamp_value:        Optional[float] = 8.0,
        generator:          Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate forecast frames by integrating the reverse process.

        Args:
            sampler: ``"pc"`` runs the reverse-time SDE with a Langevin
                corrector — stochastic, and the right choice for ensembles.
                ``"ode"`` integrates the probability-flow ODE, which is
                deterministic given the initial noise and shares the same
                marginals; use it for reproducible scoring.
            num_steps: reverse integration steps.
            corrector_steps: Langevin steps per predictor step (``pc`` only).
                The corrector is what lets a score model use far fewer solver
                steps than an ancestral chain: it re-equilibrates onto p_t
                instead of accumulating discretisation error.
            snr: Langevin step-size signal-to-noise ratio (``pc`` only).
            clamp_value: divergence guard on the latent iterate. Latents are
                unit-scaled, so this sits far outside the data distribution and
                only catches blow-ups.

        Returns:
            (B * num_samples, T_fcast, C, H, W)
        """
        if sampler not in ("pc", "ode"):
            raise ValueError(f"sampler must be 'pc' or 'ode', got {sampler!r}")

        device = context_images.device

        if num_samples > 1:
            context_images = context_images.repeat_interleave(num_samples, dim=0)
        B = context_images.shape[0]

        context_latents = self.encode_images(context_images, deterministic=True)
        encoded_context = self._encode_context(context_latents)

        shape = (
            B, num_forecast_steps,
            self.latent_channels, self.latent_size, self.latent_size,
        )
        fn = self.score_fn(encoded_context, context_latents)

        if sampler == "pc":
            x = pc_sampler(
                self.sde, fn, shape, device,
                num_steps=num_steps, corrector_steps=corrector_steps, snr=snr,
                clamp_value=clamp_value, generator=generator,
            )
        else:
            x = ode_sampler(
                self.sde, fn, shape, device,
                num_steps=num_steps, method=ode_method,
                clamp_value=clamp_value, generator=generator,
            )

        return self.decode_latents(x)

    @torch.no_grad()
    def forecast_deterministic(
        self,
        context_images:     torch.Tensor,
        num_forecast_steps: int,
        num_steps:          int = 100,
    ) -> torch.Tensor:
        """Deterministic forecast via the probability-flow ODE."""
        return self.sample(
            context_images,
            num_forecast_steps = num_forecast_steps,
            num_samples        = 1,
            sampler            = "ode",
            num_steps          = num_steps,
        )

    def extra_repr(self) -> str:
        return (
            f"latent_channels={self.latent_channels}, "
            f"latent_size={self.latent_size}, "
            f"context_length={self.context_length}, "
            f"transformer_dim={self.transformer_dim}, "
            f"sde={self.sde_name}, weighting={self.loss_weighting}, "
            f"latent_norm={self.latent_norm}"
        )
