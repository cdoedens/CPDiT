"""
Latent Diffusion Transformer for satellite image forecasting.

Training follows the two-stage LDM recipe (Rombach et al. 2022):
  Stage 1 — Train VAE alone (see vae.py / training scripts).
  Stage 2 — Freeze VAE, train the diffusion model in latent space.

Diffusion formulation:
  - Forward process : q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
  - Training target : predict the noise epsilon added at timestep t
  - Reverse process : DDPM posterior update (Ho et al. 2020)
  - Fast inference  : DDIM sampler (Song et al. 2020), configurable step count and eta

References:
  - Ho et al. 2020  — https://arxiv.org/abs/2006.11239  (DDPM)
  - Song et al. 2020 — https://arxiv.org/abs/2010.02502  (DDIM)
  - Nichol & Dhariwal 2021 — https://arxiv.org/abs/2102.09672  (cosine schedule)
  - Rombach et al. 2022 — https://arxiv.org/abs/2112.10752  (LDM)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import VariationalAutoencoder
from .transformer_backbone import TransformerBackbone


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Embeds integer diffusion timesteps into a continuous vector via sinusoidal
    encoding followed by a two-layer MLP projection.

    This gives the denoiser a smooth, high-frequency representation of t so it
    can distinguish noise levels across all 1000 diffusion steps.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Output embedding dimension. Should match latent_dim so the
                 embedding can be added directly to latent vectors.
        """
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer diffusion timesteps in [0, T-1].

        Returns:
            emb: (B, dim) timestep embeddings.
        """
        assert t.ndim == 1, f"Expected 1D timestep tensor, got shape {t.shape}"
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device).float()
            / (half - 1)
        )                                                   # (half,)
        args = t[:, None].float() * freqs[None]            # (B, half)
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)

        # Handle odd dim by zero-padding the last position
        if self.dim % 2 != 0:
            emb = F.pad(emb, (0, 1))

        return self.mlp(emb)                               # (B, dim)


# ---------------------------------------------------------------------------
# Denoising network
# ---------------------------------------------------------------------------

class DenoiserNetwork(nn.Module):
    """
    MLP-based denoising network conditioned on:
      - The noisy forecast latents x_t
      - The diffusion timestep t (via sinusoidal embedding)
      - The encoded context sequence (via mean-pooled cross-attention projection)

    Predicts the noise epsilon that was added to x_0 to produce x_t.

    Architecture:
        [x_t | t_emb | ctx_emb] → Linear → SiLU → LN → Linear → SiLU → LN → Linear
                                                                              ↓
                                                                     predicted noise
    """

    def __init__(self, latent_dim: int, context_dim: int, hidden_dim: int = 512):
        """
        Args:
            latent_dim:  Dimension of forecast latent vectors (= VAE latent_dim).
            context_dim: Dimension of transformer output (= latent_dim).
            hidden_dim:  Width of the hidden MLP layers.
        """
        super().__init__()

        self.time_emb     = SinusoidalTimestepEmbedding(latent_dim)
        self.context_proj = nn.Linear(context_dim, latent_dim)

        # Input: concatenation of [noisy_x | time_emb | context_emb] → 3 * latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        x:       torch.Tensor,   # (B, T_fcast, latent_dim) — noisy forecast latents
        t:       torch.Tensor,   # (B,)                     — integer diffusion timesteps
        context: torch.Tensor,   # (B, T_ctx,   latent_dim) — encoded context sequence
    ) -> torch.Tensor:
        """
        Returns:
            predicted_noise: (B, T_fcast, latent_dim)
        """
        T_fcast = x.size(1)

        # Timestep embedding: (B, D) → (B, T_fcast, D)
        t_emb = self.time_emb(t).unsqueeze(1).expand(-1, T_fcast, -1)

        # Context: mean-pool over context time axis → (B, D) → (B, T_fcast, D)
        ctx = self.context_proj(context.mean(dim=1))
        ctx = ctx.unsqueeze(1).expand(-1, T_fcast, -1)

        # Concatenate and denoise
        h = torch.cat([x, t_emb, ctx], dim=-1)            # (B, T_fcast, 3D)
        return self.net(h)                                 # (B, T_fcast, D)

    def extra_repr(self) -> str:
        return f"hidden_dim={self.net[0].out_features}"


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class LatentDiffusionTransformer(nn.Module):
    """
    Latent Diffusion Transformer (LDT) for probabilistic satellite image forecasting.

    Components
    ----------
    VAE            : Compresses each satellite frame to a 1-D latent vector.
    TransformerBackbone : Encodes the context latent sequence temporally.
    DenoiserNetwork : Predicts noise in the diffusion forward process,
                      conditioned on timestep and encoded context.

    Training (two stages)
    ---------------------
    Stage 1 — train VAE independently (call freeze_vae() before Stage 2).
    Stage 2 — call forward(context_images, target_images) which returns the
              diffusion noise-prediction MSE loss.

    Inference
    ---------
    Call sample() to generate forecast frames via DDIM reverse diffusion.
    """

    def __init__(
        self,
        image_channels:       int   = 5,
        image_size:           int   = 64,
        latent_dim:           int   = 256,
        hidden_dim:           int   = 256,
        num_transformer_layers: int = 4,
        num_heads:            int   = 8,
        feedforward_dim:      int   = 1024,
        num_diffusion_steps:  int   = 1000,
        denoiser_hidden_dim:  int   = 512,
        dropout:              float = 0.1,
    ):
        """
        Args:
            image_channels:         Number of satellite image channels (e.g. 3 bands).
            image_size:             Spatial size of input images (assumed square).
                                    Passed to VAE so it can compute encoder output shape.
            latent_dim:             Dimension of the VAE latent space.
            num_transformer_layers: Depth of the temporal transformer.
            num_heads:              Number of attention heads in the transformer.
            feedforward_dim:        FFN width inside each transformer layer.
            num_diffusion_steps:    Total diffusion timesteps T (used for scheduling).
            denoiser_hidden_dim:    Hidden width of the DenoiserNetwork MLP.
            dropout:                Dropout rate in the transformer.
        """
        super().__init__()

        self.image_channels      = image_channels
        self.latent_dim          = latent_dim
        self.num_diffusion_steps = num_diffusion_steps

        # ------------------------------------------------------------------ #
        # Sub-modules                                                          #
        # ------------------------------------------------------------------ #

        self.vae = VariationalAutoencoder(
            image_channels=image_channels,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            image_size=image_size,
        )

        self.transformer = TransformerBackbone(
            latent_dim=latent_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

        self.denoiser = DenoiserNetwork(
            latent_dim=latent_dim,
            context_dim=latent_dim,
            hidden_dim=denoiser_hidden_dim,
        )

        # ------------------------------------------------------------------ #
        # Diffusion noise schedule (cosine, Nichol & Dhariwal 2021)           #
        # All schedule tensors are registered as buffers so they move with    #
        # the model across devices and are saved in checkpoints.              #
        # ------------------------------------------------------------------ #

        betas = self._cosine_beta_schedule(num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]]
        )

        self.register_buffer("betas",                    betas)
        self.register_buffer("alphas_cumprod",           alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev",      alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod",      torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - alphas_cumprod))

    # ------------------------------------------------------------------ #
    # Noise schedule                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine variance schedule (Nichol & Dhariwal 2021).

        Keeps alpha_bar from dropping to zero too quickly at high t,
        which preserves more signal and produces better sample quality
        than the original linear schedule.

        Args:
            timesteps: Total number of diffusion steps T.
            s:         Small offset to prevent beta from being too small
                       near t=0 (default 0.008 from the paper).

        Returns:
            betas: (T,) noise variance schedule, clamped to [1e-5, 0.999].
        """
        steps     = timesteps + 1
        t         = torch.linspace(0, timesteps, steps) / timesteps
        alpha_bar = torch.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]              # normalise: alpha_bar[0] = 1
        betas     = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return torch.clamp(betas, min=1e-5, max=0.999)

    # ------------------------------------------------------------------ #
    # VAE helpers                                                          #
    # ------------------------------------------------------------------ #

    def freeze_vae(self) -> None:
        """
        Freeze all VAE parameters and set it to eval mode.

        Call this before Stage 2 training so the latent space remains
        stable while the diffusion model learns to denoise within it.
        """
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

    def unfreeze_vae(self) -> None:
        """Re-enable VAE training (e.g. for joint fine-tuning)."""
        for param in self.vae.parameters():
            param.requires_grad = True
        self.vae.train()

    def encode_images(
        self,
        images:        torch.Tensor,  # (B, T, C, H, W)
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Encode a sequence of satellite frames to latent vectors.

        Args:
            images:        (B, T, C, H, W) float tensor of satellite images.
            deterministic: If True  → return VAE posterior mean mu (no sampling).
                           If False → sample z via reparameterisation trick.
                           Use True for diffusion training / inference.
                           Use False during VAE Stage 1 training.

        Returns:
            latents: (B, T, latent_dim)
        """
        assert images.ndim == 5, \
            f"Expected 5D input (B, T, C, H, W), got shape {images.shape}"
        assert images.size(2) == self.image_channels, \
            f"Expected {self.image_channels} image channels, got {images.size(2)}"

        B, T = images.shape[:2]
        images_flat = images.view(B * T, *images.shape[2:])   # (B*T, C, H, W)

        mu, logvar = self.vae.encode(images_flat)

        if deterministic:
            z = mu
        else:
            z = self.vae.reparameterize(mu, logvar)

        return z.view(B, T, self.latent_dim)                  # (B, T, D)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode a sequence of latent vectors back to satellite frames.

        Args:
            latents: (B, T, latent_dim)

        Returns:
            images: (B, T, C, H, W)
        """
        assert latents.ndim == 3, \
            f"Expected 3D input (B, T, D), got shape {latents.shape}"

        B, T = latents.shape[:2]
        latents_flat = latents.view(B * T, self.latent_dim)   # (B*T, D)
        images_flat  = self.vae.decode(latents_flat)          # (B*T, C, H, W)
        return images_flat.view(B, T, *images_flat.shape[1:]) # (B, T, C, H, W)

    # ------------------------------------------------------------------ #
    # Forward diffusion (noise addition)                                   #
    # ------------------------------------------------------------------ #

    def add_noise(
        self,
        x:         torch.Tensor,  # (B, T_fcast, D) — clean target latents
        timesteps: torch.Tensor,  # (B,)            — integer timestep per sample
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample noisy latents x_t from the forward diffusion process:
            x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * epsilon
            epsilon ~ N(0, I)

        Args:
            x:         Clean latent codes x_0.
            timesteps: Per-sample diffusion timestep t in [0, T-1].

        Returns:
            noisy:  x_t — noisy version of x.
            noise:  epsilon — the noise that was added (training target).
        """
        noise = torch.randn_like(x)

        # Extract per-sample schedule values and reshape for broadcasting
        # sqrt_alphas_cumprod[t]: (B,) → (B, 1, 1)
        sqrt_alpha_bar     = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_one_minus_bar = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)

        noisy = sqrt_alpha_bar * x + sqrt_one_minus_bar * noise
        return noisy, noise

    # ------------------------------------------------------------------ #
    # Training forward pass                                                #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        context_images: torch.Tensor,          # (B, T_ctx,   C, H, W)
        target_images:  torch.Tensor,          # (B, T_fcast, C, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Diffusion training step.

        Randomly samples a diffusion timestep t for each batch element,
        corrupts the target latents to x_t, and trains the denoiser to
        recover the added noise — the standard DDPM training objective.

        Args:
            context_images: Past satellite frames (encoder input).
            target_images:  Future satellite frames (diffusion target).

        Returns:
            loss:            Scalar MSE between true and predicted noise.
            encoded_context: (B, T_ctx, D) transformer output, returned for
                             optional auxiliary losses or logging.
        """
        B = context_images.shape[0]

        # 1. Encode both sequences to latent space (deterministic — VAE frozen in Stage 2)
        context_latents = self.encode_images(context_images, deterministic=True)
        target_latents  = self.encode_images(target_images,  deterministic=True)

        # 2. Encode context sequence through temporal transformer
        encoded_context = self.transformer(context_latents)   # (B, T_ctx, D)

        # 3. Sample a random diffusion timestep per batch element
        t = torch.randint(
            0, self.num_diffusion_steps, (B,), device=context_images.device
        )

        # 4. Corrupt target latents to x_t
        noisy_targets, noise = self.add_noise(target_latents, t)

        # 5. Predict the noise using the denoiser
        predicted_noise = self.denoiser(noisy_targets, t, encoded_context)

        # 6. Compute noise-prediction MSE loss (DDPM objective)
        loss = F.mse_loss(predicted_noise, noise)

        return loss, encoded_context

    # ------------------------------------------------------------------ #
    # Inference: DDIM reverse diffusion                                    #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def sample(
        self,
        context_images:     torch.Tensor,
        num_forecast_steps: int,
        num_samples:        int   = 1,
        ddim_steps:         int   = 50,
        eta:                float = 0.0,
        clamp_latents:      bool  = True,
    ) -> torch.Tensor:
        B = context_images.shape[0]
        device = context_images.device          # ← capture once, use everywhere
    
        if num_samples > 1:
            context_images = context_images.repeat_interleave(num_samples, dim=0)
            B = context_images.shape[0]
    
        context_latents = self.encode_images(context_images, deterministic=True)
        encoded_context = self.transformer(context_latents)
    
        # Build DDIM subsequence
        step_ratio = max(self.num_diffusion_steps // ddim_steps, 1)
        ddim_timesteps = list(
            reversed(range(0, self.num_diffusion_steps, step_ratio))
        )[:ddim_steps]
    
        x = torch.randn(
            B, num_forecast_steps, self.latent_dim, device=device
        )
    
        for i, t_val in enumerate(ddim_timesteps):
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
    
            pred_noise = self.denoiser(x, t_tensor, encoded_context)
    
            alpha_bar = self.alphas_cumprod[t_val]          # scalar tensor, on device
    
            # FIX 1: keep alpha_bar_prev on the correct device
            t_prev = ddim_timesteps[i + 1] if i + 1 < len(ddim_timesteps) else -1
            if t_prev >= 0:
                alpha_bar_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)   # ← device kwarg
    
            # FIX 2: guard the division so sqrt never receives a negative value
            sqrt_alpha_bar = torch.sqrt(alpha_bar).clamp(min=1e-8)
    
            # FIX 3: clamp pred_x0 BEFORE the division blows up at high t
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar) * pred_noise) / sqrt_alpha_bar
            if clamp_latents:
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
    
            # Sigma: clamp the radicand to ≥ 0 to guard against schedule rounding
            sigma_sq_arg = (
                (1.0 - alpha_bar_prev)
                / (1.0 - alpha_bar).clamp(min=1e-8)
                * (1.0 - alpha_bar / alpha_bar_prev.clamp(min=1e-8))
            ).clamp(min=0.0)                                         # ← FIX 2
            sigma = eta * torch.sqrt(sigma_sq_arg)
    
            # Direction term: clamp radicand to ≥ 0 for the same reason
            dir_sq_arg = (1.0 - alpha_bar_prev - sigma ** 2).clamp(min=0.0)
            direction = torch.sqrt(dir_sq_arg) * pred_noise
    
            x = (
                torch.sqrt(alpha_bar_prev) * pred_x0
                + direction
                + sigma * torch.randn_like(x)
            )
    
        return self.decode_latents(x)


    # ------------------------------------------------------------------ #
    # Convenience: deterministic forecast (no diffusion noise)             #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def forecast_deterministic(
        self,
        context_images:     torch.Tensor,   # (B, T_ctx, C, H, W)
        num_forecast_steps: int,
    ) -> torch.Tensor:
        """
        Generate a single deterministic forecast by running DDIM with eta=0
        and a reduced step count.

        Equivalent to calling sample(..., num_samples=1, eta=0.0).
        Useful for fast evaluation and benchmarking.

        Returns:
            forecast: (B, T_fcast, C, H, W)
        """
        return self.sample(
            context_images,
            num_forecast_steps=num_forecast_steps,
            num_samples=1,
            ddim_steps=50,
            eta=0.0,
        )

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.latent_dim}, "
            f"num_diffusion_steps={self.num_diffusion_steps}"
        )
