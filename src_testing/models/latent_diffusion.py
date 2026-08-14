"""
Latent Diffusion Transformer for satellite image forecasting.

Training follows the two-stage LDM recipe (Rombach et al. 2022):
  Stage 1 — Train VAE alone (see vae.py / training scripts).
  Stage 2 — Freeze VAE, train the diffusion model in latent space.

Latent space:
  The VAE now produces spatial latent maps of shape
  (latent_channels, H/8, W/8) rather than flat vectors.
  For image_size=256 this gives (4, 32, 32) per frame.

  The diffusion model operates entirely on these spatial maps.
  The transformer encodes the context sequence by treating each
  frame's flattened latent map as a token sequence.

Diffusion formulation:
  - Forward process : q(x_t | x_0) = N(sqrt(ā_t)*x_0, (1-ā_t)*I)
  - Training target : predict noise epsilon added at timestep t
  - Reverse process : DDPM posterior update (Ho et al. 2020)
  - Fast inference  : DDIM sampler (Song et al. 2020)

References:
  - Ho et al. 2020           — https://arxiv.org/abs/2006.11239
  - Song et al. 2020         — https://arxiv.org/abs/2010.02502
  - Nichol & Dhariwal 2021   — https://arxiv.org/abs/2102.09672
  - Rombach et al. 2022      — https://arxiv.org/abs/2112.10752
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
    Sinusoidal timestep embedding followed by a two-layer MLP.

    Output dim is chosen to match the denoiser's internal channel width
    so the embedding can be broadcast-added to feature maps.
    """

    def __init__(self, dim: int):
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
            t: (B,) integer diffusion timesteps.
        Returns:
            emb: (B, dim)
        """
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device).float()
            / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 != 0:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Spatial denoiser (U-Net style)
# ---------------------------------------------------------------------------

class ResBlockSpatial(nn.Module):
    """
    Spatial residual block conditioned on a timestep + context embedding.

    The conditioning vector is projected to (2 * channels) and used for
    FiLM-style scale/shift modulation after the first GroupNorm, which is
    more expressive than simple addition.
    """

    def __init__(self, channels: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act   = nn.SiLU()

        # FiLM conditioning: project cond → scale and shift for each channel
        self.cond_proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, C, H, W)
            cond: (B, cond_dim)
        Returns:
            (B, C, H, W)
        """
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)   # each (B, C)
        scale = scale[:, :, None, None]                         # (B, C, 1, 1)
        shift = shift[:, :, None, None]

        h = self.norm1(x)
        h = h * (1.0 + scale) + shift                          # FiLM modulation
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return x + h


class DenoiserNetwork(nn.Module):
    """
    Lightweight U-Net denoiser that operates on spatial latent maps.

    Input:  (B, T_fcast, latent_channels, Hl, Wl)  — noisy forecast latents
    Output: (B, T_fcast, latent_channels, Hl, Wl)  — predicted noise

    Each forecast frame is denoised independently but conditioned on the
    same (timestep, context) embedding, so the model is frame-agnostic
    and generalises to any forecast length.

    Architecture (per frame):
        entry conv
        → down1 (D)   → down2 (2D)  → bottleneck (4D)
        → up1   (2D)  → up2   (D)
        → exit conv

    All ResBlocks are FiLM-conditioned on [t_emb + ctx_emb].
    """

    def __init__(
        self,
        latent_channels: int,
        latent_size:     int,
        context_dim:     int,
        hidden_dim:      int = 128,
    ):
        """
        Args:
            latent_channels: Channels in the VAE latent map (e.g. 4).
            latent_size:     Spatial size of the latent map (e.g. 32 for 256px images).
            context_dim:     Dimension of the transformer context output token.
            hidden_dim:      Base channel width of the U-Net.
        """
        super().__init__()

        D        = hidden_dim
        cond_dim = D * 4   # conditioning vector width

        # ------------------------------------------------------------------ #
        # Conditioning: timestep + context → single conditioning vector       #
        # ------------------------------------------------------------------ #
        self.time_emb    = SinusoidalTimestepEmbedding(D)
        self.ctx_proj    = nn.Linear(context_dim, D)
        self.cond_fusion = nn.Sequential(
            nn.Linear(D * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # ------------------------------------------------------------------ #
        # U-Net encoder                                                        #
        # ------------------------------------------------------------------ #
        self.entry  = nn.Conv2d(latent_channels, D, kernel_size=3, padding=1)

        self.down1  = nn.Conv2d(D,     D * 2, kernel_size=4, stride=2, padding=1)
        self.res_d1 = ResBlockSpatial(D * 2, cond_dim)

        self.down2  = nn.Conv2d(D * 2, D * 4, kernel_size=4, stride=2, padding=1)
        self.res_d2 = ResBlockSpatial(D * 4, cond_dim)

        # ------------------------------------------------------------------ #
        # Bottleneck                                                           #
        # ------------------------------------------------------------------ #
        self.res_mid1 = ResBlockSpatial(D * 4, cond_dim)
        self.res_mid2 = ResBlockSpatial(D * 4, cond_dim)

        # ------------------------------------------------------------------ #
        # U-Net decoder (with skip connections)                               #
        # ------------------------------------------------------------------ #
        self.up1    = nn.ConvTranspose2d(D * 4, D * 2, kernel_size=4, stride=2, padding=1)
        self.res_u1 = ResBlockSpatial(D * 2, cond_dim)   # input already skip-summed

        self.up2    = nn.ConvTranspose2d(D * 2, D,     kernel_size=4, stride=2, padding=1)
        self.res_u2 = ResBlockSpatial(D, cond_dim)

        # ------------------------------------------------------------------ #
        # Exit                                                                 #
        # ------------------------------------------------------------------ #
        self.exit = nn.Sequential(
            nn.GroupNorm(8, D),
            nn.SiLU(),
            nn.Conv2d(D, latent_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x:       torch.Tensor,   # (B, T_fcast, latent_channels, Hl, Wl)
        t:       torch.Tensor,   # (B,)
        context: torch.Tensor,   # (B, T_ctx, context_dim)
    ) -> torch.Tensor:
        """
        Returns:
            predicted_noise: (B, T_fcast, latent_channels, Hl, Wl)
        """
        B, T_fcast, C, Hl, Wl = x.shape

        # Build conditioning vector: mean-pool context, fuse with time emb
        t_emb   = self.time_emb(t)                          # (B, D)
        ctx_emb = self.ctx_proj(context.mean(dim=1))        # (B, D)
        cond    = self.cond_fusion(
            torch.cat([t_emb, ctx_emb], dim=-1)
        )                                                    # (B, 4D)

        # Process each forecast frame independently
        x_flat = x.view(B * T_fcast, C, Hl, Wl)

        # Expand cond to match flattened batch
        cond_exp = cond.unsqueeze(1).expand(-1, T_fcast, -1)  # (B, T, 4D)
        cond_exp = cond_exp.reshape(B * T_fcast, -1)           # (B*T, 4D)

        # U-Net forward
        h0 = self.entry(x_flat)                              # (B*T, D,  Hl,   Wl  )

        h1 = self.down1(h0)                                  # (B*T, 2D, Hl/2, Wl/2)
        h1 = self.res_d1(h1, cond_exp)

        h2 = self.down2(h1)                                  # (B*T, 4D, Hl/4, Wl/4)
        h2 = self.res_d2(h2, cond_exp)

        h  = self.res_mid1(h2, cond_exp)
        h  = self.res_mid2(h,  cond_exp)

        h  = self.up1(h) + h1                               # skip from down1
        h  = self.res_u1(h, cond_exp)

        h  = self.up2(h) + h0                               # skip from entry
        h  = self.res_u2(h, cond_exp)

        out = self.exit(h)                                   # (B*T, C, Hl, Wl)
        return out.view(B, T_fcast, C, Hl, Wl)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class LatentDiffusionTransformer(nn.Module):
    """
    Latent Diffusion Transformer (LDT) for probabilistic satellite image forecasting.

    Components
    ----------
    VAE                  : Compresses (C, H, W) → (latent_channels, H/8, W/8).
    TransformerBackbone  : Encodes the context latent sequence temporally.
                           Each frame's latent map is flattened to a 1-D token
                           via a linear projection before being passed to the
                           transformer, then unprojected after.
    DenoiserNetwork      : Spatial U-Net that predicts noise in the diffusion
                           forward process, conditioned on timestep and context.
    """

    def __init__(
        self,
        image_channels:         int   = 3,
        image_size:             int   = 256,
        latent_channels:        int   = 4,
        vae_hidden_dim:         int   = 128,
        num_transformer_layers: int   = 4,
        num_heads:              int   = 8,
        feedforward_dim:        int   = 1024,
        transformer_dim:        int   = 512,
        num_diffusion_steps:    int   = 1000,
        denoiser_hidden_dim:    int   = 128,
        dropout:                float = 0.1,
    ):
        """
        Args:
            image_channels:         Satellite image channels (e.g. 3 bands).
            image_size:             Spatial size of input images (must be div by 8).
            latent_channels:        VAE latent map channels (4 is standard).
            vae_hidden_dim:         Base channel width of the VAE conv stack.
            num_transformer_layers: Depth of the temporal transformer.
            num_heads:              Attention heads in the transformer.
            feedforward_dim:        FFN width inside each transformer layer.
            transformer_dim:        Token dimension fed into the transformer.
                                    The flattened latent map is projected to this dim.
            num_diffusion_steps:    Total diffusion timesteps T.
            denoiser_hidden_dim:    Base channel width of the U-Net denoiser.
            dropout:                Dropout rate in the transformer.
        """
        super().__init__()

        self.image_channels      = image_channels
        self.latent_channels     = latent_channels
        self.latent_size         = image_size // 8   # spatial size of latent map
        self.num_diffusion_steps = num_diffusion_steps
        self.transformer_dim     = transformer_dim

        # Flat size of one latent frame: latent_channels * (H/8) * (W/8)
        self._latent_flat_dim = latent_channels * self.latent_size * self.latent_size

        # ------------------------------------------------------------------ #
        # Sub-modules                                                          #
        # ------------------------------------------------------------------ #

        self.vae = VariationalAutoencoder(
            image_channels  = image_channels,
            latent_channels = latent_channels,
            hidden_dim      = vae_hidden_dim,
            image_size      = image_size,
        )

        # Project flat latent → transformer token dim and back
        self.latent_to_token  = nn.Linear(self._latent_flat_dim, transformer_dim)
        self.token_to_latent  = nn.Linear(transformer_dim, self._latent_flat_dim)

        self.transformer = TransformerBackbone(
            latent_dim      = transformer_dim,
            num_layers      = num_transformer_layers,
            num_heads       = num_heads,
            feedforward_dim = feedforward_dim,
            dropout         = dropout,
        )

        self.denoiser = DenoiserNetwork(
            latent_channels = latent_channels,
            latent_size     = self.latent_size,
            context_dim     = transformer_dim,
            hidden_dim      = denoiser_hidden_dim,
        )

        # ------------------------------------------------------------------ #
        # Cosine noise schedule (Nichol & Dhariwal 2021)                      #
        # ------------------------------------------------------------------ #

        betas              = self._cosine_beta_schedule(num_diffusion_steps)
        alphas             = 1.0 - betas
        alphas_cumprod     = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]]
        )

        self.register_buffer("betas",                         betas)
        self.register_buffer("alphas_cumprod",                alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev",           alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod",           torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    # ------------------------------------------------------------------ #
    # Noise schedule                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps     = timesteps + 1
        t         = torch.linspace(0, timesteps, steps) / timesteps
        alpha_bar = torch.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas     = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return torch.clamp(betas, min=1e-5, max=0.999)

    # ------------------------------------------------------------------ #
    # VAE helpers                                                          #
    # ------------------------------------------------------------------ #

    def freeze_vae(self) -> None:
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

    def unfreeze_vae(self) -> None:
        for p in self.vae.parameters():
            p.requires_grad = True
        self.vae.train()

    def encode_images(
        self,
        images:        torch.Tensor,   # (B, T, C, H, W)
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Encode a frame sequence to spatial latent maps.

        Returns:
            latents: (B, T, latent_channels, Hl, Wl)
        """
        assert images.ndim == 5, \
            f"Expected (B, T, C, H, W), got {images.shape}"
        B, T = images.shape[:2]
        flat = images.view(B * T, *images.shape[2:])

        mu, logvar = self.vae.encode(flat)
        z = mu if deterministic else self.vae.reparameterize(mu, logvar)

        Hl, Wl = z.shape[-2], z.shape[-1]
        return z.view(B, T, self.latent_channels, Hl, Wl)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode spatial latent maps to image frames.

        Args:
            latents: (B, T, latent_channels, Hl, Wl)
        Returns:
            images:  (B, T, C, H, W)
        """
        assert latents.ndim == 5, \
            f"Expected (B, T, LC, Hl, Wl), got {latents.shape}"
        B, T = latents.shape[:2]
        flat    = latents.view(B * T, *latents.shape[2:])
        decoded = self.vae.decode(flat)
        return decoded.view(B, T, *decoded.shape[1:])

    def _encode_context(self, context_latents: torch.Tensor) -> torch.Tensor:
        """
        Project spatial latent maps → transformer tokens → encode temporally.

        Args:
            context_latents: (B, T_ctx, latent_channels, Hl, Wl)
        Returns:
            encoded:         (B, T_ctx, transformer_dim)
        """
        B, T = context_latents.shape[:2]
        # Flatten spatial dims: (B, T, LC*Hl*Wl)
        tokens = context_latents.view(B, T, -1)
        # Project to transformer dim: (B, T, transformer_dim)
        tokens = self.latent_to_token(tokens)
        # Temporal encoding
        return self.transformer(tokens)

    # ------------------------------------------------------------------ #
    # Forward diffusion                                                    #
    # ------------------------------------------------------------------ #

    def add_noise(
        self,
        x:         torch.Tensor,   # (B, T, latent_channels, Hl, Wl)
        timesteps: torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x)

        # Reshape schedule values for broadcasting over (T, C, H, W)
        sqrt_ab     = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        sqrt_one_ab = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)

        noisy = sqrt_ab * x + sqrt_one_ab * noise
        return noisy, noise

    # ------------------------------------------------------------------ #
    # Training forward pass                                                #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        context_images: torch.Tensor,   # (B, T_ctx,   C, H, W)
        target_images:  torch.Tensor,   # (B, T_fcast, C, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        DDPM training step.

        Returns:
            loss:            Scalar noise-prediction MSE.
            encoded_context: (B, T_ctx, transformer_dim) for logging.
        """
        B      = context_images.shape[0]
        device = context_images.device

        # 1. Encode to spatial latents
        context_latents = self.encode_images(context_images, deterministic=True)
        target_latents  = self.encode_images(target_images,  deterministic=True)

        # 2. Temporally encode context
        encoded_context = self._encode_context(context_latents)

        # 3. Sample diffusion timestep
        t = torch.randint(0, self.num_diffusion_steps, (B,), device=device)

        # 4. Corrupt target latents
        noisy_targets, noise = self.add_noise(target_latents, t)

        # 5. Predict noise
        predicted_noise = self.denoiser(noisy_targets, t, encoded_context)

        # 6. DDPM loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss, encoded_context

    @torch.no_grad()
    def forward_eval(
        self,
        context_images: torch.Tensor,   # (B, T_ctx,   C, H, W)
        target_images:  torch.Tensor,   # (B, T_fcast, C, H, W)
    ) -> dict[str, torch.Tensor]:
        """
        Evaluation forward pass. Returns irradiance-specific pixel-space metrics
        alongside the standard latent noise loss.
    
        The predicted clean image is estimated from the single-step x0 prediction
        (DDPM eq. 15) rather than running the full reverse chain — this is fast
        and gives a meaningful signal about denoiser quality at each noise level.
    
        Returns a dict with keys:
            latent_loss       : standard noise-prediction MSE (all channels)
            irradiance_mse    : pixel-space MSE on irradiance channel only
            irradiance_mae    : pixel-space MAE on irradiance channel only
        """
        B      = context_images.shape[0]
        device = context_images.device
    
        context_latents = self.encode_images(context_images, deterministic=True)
        target_latents  = self.encode_images(target_images,  deterministic=True)
        encoded_context = self._encode_context(context_latents)
    
        t = torch.randint(0, self.num_diffusion_steps, (B,), device=device)
    
        noisy_targets, noise = self.add_noise(target_latents, t)
        predicted_noise      = self.denoiser(noisy_targets, t, encoded_context)
    
        # Standard latent loss — same as training
        latent_loss = F.mse_loss(predicted_noise, noise)
    
        # Recover predicted x0 from noise prediction (DDPM eq. 15)
        sqrt_ab     = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1, 1)
        sqrt_one_ab = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1, 1)
        pred_x0_latent = (
            (noisy_targets - sqrt_one_ab * predicted_noise)
            / sqrt_ab.clamp(min=1e-8)
        )
    
        # Decode to pixel space — VAE decoder is frozen so this is cheap
        pred_images = self.decode_latents(pred_x0_latent)   # (B, T, C, H, W)
    
        # Score irradiance channel only
        pred_irr   = pred_images[:, :, 0:1, :, :]
        target_irr = target_images[:, :, 0:1, :, :]
    
        return {
            "latent_loss":    latent_loss,
            "irradiance_mse": F.mse_loss(pred_irr, target_irr),
            "irradiance_mae": F.l1_loss(pred_irr, target_irr),
        }


    # ------------------------------------------------------------------ #
    # Inference: DDIM reverse diffusion                                    #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def sample(
        self,
        context_images:     torch.Tensor,   # (B, T_ctx, C, H, W)
        num_forecast_steps: int,
        num_samples:        int   = 1,
        ddim_steps:         int   = 50,
        eta:                float = 0.0,
        clamp_latents:      bool  = True,
    ) -> torch.Tensor:
        """
        Generate forecast frames via DDIM reverse diffusion.

        Returns:
            forecast: (B * num_samples, T_fcast, C, H, W)
        """
        B      = context_images.shape[0]
        device = context_images.device

        if num_samples > 1:
            context_images = context_images.repeat_interleave(num_samples, dim=0)
            B = context_images.shape[0]

        context_latents = self.encode_images(context_images, deterministic=True)
        encoded_context = self._encode_context(context_latents)

        # DDIM timestep subsequence
        step_ratio     = max(self.num_diffusion_steps // ddim_steps, 1)
        ddim_timesteps = list(
            reversed(range(0, self.num_diffusion_steps, step_ratio))
        )[:ddim_steps]

        # Start from pure noise: (B, T_fcast, LC, Hl, Wl)
        x = torch.randn(
            B, num_forecast_steps,
            self.latent_channels, self.latent_size, self.latent_size,
            device=device,
        )

        for i, t_val in enumerate(ddim_timesteps):
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)

            pred_noise = self.denoiser(x, t_tensor, encoded_context)

            alpha_bar = self.alphas_cumprod[t_val]

            t_prev = ddim_timesteps[i + 1] if i + 1 < len(ddim_timesteps) else -1
            alpha_bar_prev = (
                self.alphas_cumprod[t_prev]
                if t_prev >= 0
                else torch.tensor(1.0, device=device)
            )

            sqrt_ab  = torch.sqrt(alpha_bar).clamp(min=1e-8)
            pred_x0  = (x - torch.sqrt(1.0 - alpha_bar) * pred_noise) / sqrt_ab
            if clamp_latents:
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            sigma_sq_arg = (
                (1.0 - alpha_bar_prev)
                / (1.0 - alpha_bar).clamp(min=1e-8)
                * (1.0 - alpha_bar / alpha_bar_prev.clamp(min=1e-8))
            ).clamp(min=0.0)
            sigma     = eta * torch.sqrt(sigma_sq_arg)
            direction = torch.sqrt(
                (1.0 - alpha_bar_prev - sigma ** 2).clamp(min=0.0)
            ) * pred_noise

            x = (
                torch.sqrt(alpha_bar_prev) * pred_x0
                + direction
                + sigma * torch.randn_like(x)
            )

        return self.decode_latents(x)

    # ------------------------------------------------------------------ #
    # Deterministic forecast                                               #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def forecast_deterministic(
        self,
        context_images:     torch.Tensor,
        num_forecast_steps: int,
    ) -> torch.Tensor:
        return self.sample(
            context_images,
            num_forecast_steps = num_forecast_steps,
            num_samples        = 1,
            ddim_steps         = 50,
            eta                = 0.0,
        )

    def extra_repr(self) -> str:
        return (
            f"latent_channels={self.latent_channels}, "
            f"latent_size={self.latent_size}, "
            f"transformer_dim={self.transformer_dim}, "
            f"num_diffusion_steps={self.num_diffusion_steps}"
        )
