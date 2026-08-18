"""
Latent Diffusion Transformer for satellite image forecasting.

Training follows the two-stage LDM recipe (Rombach et al. 2022):
  Stage 1 — Train VAE alone (see vae.py / training scripts).
  Stage 2 — Freeze VAE, train the diffusion model in latent space.

Latent space:
  The VAE produces spatial latent maps of shape
  (latent_channels, H/8, W/8) rather than flat vectors.
  For image_size=256 this gives (4, 32, 32) per frame.

  The diffusion model operates entirely on these spatial maps.
  The transformer encodes the context sequence; each frame's
  latent map is encoded by a small CNN stem into a single token
  before being passed to the transformer.

Input channels:
  Channel 0 — surface solar irradiance       (satellite)
  Channel 1 — solar elevation angle          (geometric)
  Channel 2 — atmospheric instability index  (reanalysis / BARRA)

  Channels 1 and 2 are treated as auxiliary BARRA/reanalysis context
  and injected into the transformer via cross-attention, keeping
  the irradiance latent stream and auxiliary conditioning separate.

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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import VariationalAutoencoder
from .transformer_backbone import TransformerBackbone, TimestepEmbedder


# ---------------------------------------------------------------------------
# Spatial frame encoder
# ---------------------------------------------------------------------------

class SpatialFrameEncoder(nn.Module):
    """
    Encodes a single spatial latent map (LC, Hl, Wl) into a 1-D token of
    size `token_dim` using a small CNN stem followed by global average pooling.

    This replaces the flat linear projection (latent_to_token) from the
    previous version. The CNN preserves local spatial structure through
    its receptive field before collapsing to a token, whereas a flat linear
    layer treats every pixel independently with no spatial inductive bias.

    Architecture:
        Conv(LC  → D/2, 3×3)  + SiLU
        Conv(D/2 → D,   3×3, stride 2) + SiLU   ← halves spatial dims
        Conv(D   → D,   3×3)  + SiLU
        GlobalAvgPool → (B, D)
    """

    def __init__(self, latent_channels: int, token_dim: int):
        super().__init__()
        D = token_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(latent_channels, D // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(D // 2, D, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(D, D, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # (B, D, 1, 1) → (B, D)
        self.norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, LC, Hl, Wl)
        Returns:
            token: (B, D)
        """
        h = self.cnn(x)
        h = self.pool(h).flatten(1)   # (B, D)
        return self.norm(h)


# ---------------------------------------------------------------------------
# Cross-attention context aggregator (for denoiser ResBlocks)
# ---------------------------------------------------------------------------

class ContextCrossAttention(nn.Module):
    """
    Aggregates a variable-length context sequence (B, T, D) into a single
    conditioning vector (B, D) via cross-attention with a learned query.

    Used inside the denoiser to replace mean-pooling, preserving temporal
    structure from the transformer-encoded context.

    A single learned query attends over all T context tokens, producing a
    weighted summary that the denoiser can condition on per-frame.
    """

    def __init__(self, context_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.query  = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.proj_kv = nn.Linear(context_dim, hidden_dim * 2, bias=False)
        self.attn   = nn.MultiheadAttention(
            embed_dim  = hidden_dim,
            num_heads  = num_heads,
            batch_first = True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        context: torch.Tensor,                        # (B, T, context_dim)
        context_padding_mask: torch.Tensor | None = None,  # (B, T)
    ) -> torch.Tensor:
        """
        Returns:
            aggregated: (B, hidden_dim)
        """
        B = context.size(0)
        k, v   = self.proj_kv(context).chunk(2, dim=-1)   # each (B, T, D)
        q      = self.query.expand(B, -1, -1)              # (B, 1, D)
        out, _ = self.attn(
            q, k, v,
            key_padding_mask = context_padding_mask,
            need_weights     = False,
        )                                                  # (B, 1, D)
        return self.norm(out.squeeze(1))                   # (B, D)


# ---------------------------------------------------------------------------
# Spatial denoiser (U-Net style)
# ---------------------------------------------------------------------------

class ResBlockSpatial(nn.Module):
    """
    Spatial residual block with two conditioning pathways:

    1. FiLM modulation from the timestep embedding — scale/shift applied
       after the first GroupNorm, controlling the overall activation magnitude
       at each noise level.

    2. Cross-attention over the transformer context sequence — a learned
       single-query cross-attention aggregates the full temporal context into
       a per-channel additive bias after the second GroupNorm. This lets each
       spatial block selectively attend to relevant context frames rather than
       receiving a fixed mean-pooled summary.
    """

    def __init__(
        self,
        channels:    int,
        cond_dim:    int,
        context_dim: int,
        num_groups:  int = 8,
        num_heads:   int = 4,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act   = nn.SiLU()

        # Pathway 1: FiLM from timestep
        self.cond_proj = nn.Linear(cond_dim, channels * 2)

        # Pathway 2: cross-attention over context sequence
        self.ctx_attn = ContextCrossAttention(context_dim, channels, num_heads)
        self.ctx_proj = nn.Linear(channels, channels)

    def forward(
        self,
        x:       torch.Tensor,   # (B, C, H, W)
        cond:    torch.Tensor,   # (B, cond_dim)   — timestep conditioning
        context: torch.Tensor,   # (B, T, context_dim) — transformer output
    ) -> torch.Tensor:
        # --- FiLM modulation (timestep) ---
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)   # each (B, C)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm1(x)
        h = h * (1.0 + scale) + shift
        h = self.act(h)
        h = self.conv1(h)

        # --- Cross-attention conditioning (context) ---
        ctx_vec = self.ctx_attn(context)                       # (B, C)
        ctx_bias = self.ctx_proj(ctx_vec)[:, :, None, None]    # (B, C, 1, 1)

        h = self.norm2(h)
        h = self.act(h + ctx_bias)
        h = self.conv2(h)
        return x + h


class DenoiserNetwork(nn.Module):
    """
    Lightweight U-Net denoiser that operates on spatial latent maps.

    Input:  (B, T_fcast, latent_channels, Hl, Wl)  — noisy forecast latents
    Output: (B, T_fcast, latent_channels, Hl, Wl)  — predicted noise

    Each forecast frame is denoised independently but conditioned on:
      - A sinusoidal + MLP timestep embedding (FiLM into every ResBlock).
      - The full transformer context sequence (cross-attention in every ResBlock).

    Architecture (per frame):
        entry conv
        → down1 (D)   → down2 (2D)  → bottleneck (4D)
        → up1   (2D)  → up2   (D)
        → exit conv
    """

    def __init__(
        self,
        latent_channels: int,
        latent_size:     int,
        context_dim:     int,
        hidden_dim:      int = 128,
        num_heads:       int = 4,
    ):
        super().__init__()

        D        = hidden_dim
        cond_dim = D * 4

        # ------------------------------------------------------------------ #
        # Timestep conditioning                                                #
        # ------------------------------------------------------------------ #
        # Re-uses the shared TimestepEmbedder from transformer_backbone.py
        # rather than duplicating a second sinusoidal embedding implementation.
        self.time_emb    = TimestepEmbedder(D)
        self.cond_expand = nn.Sequential(
            nn.Linear(D, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # ------------------------------------------------------------------ #
        # U-Net encoder                                                        #
        # ------------------------------------------------------------------ #
        self.entry  = nn.Conv2d(latent_channels, D, kernel_size=3, padding=1)

        self.down1  = nn.Conv2d(D,     D * 2, kernel_size=4, stride=2, padding=1)
        self.res_d1 = ResBlockSpatial(D * 2, cond_dim, context_dim, num_heads=num_heads)

        self.down2  = nn.Conv2d(D * 2, D * 4, kernel_size=4, stride=2, padding=1)
        self.res_d2 = ResBlockSpatial(D * 4, cond_dim, context_dim, num_heads=num_heads)

        # ------------------------------------------------------------------ #
        # Bottleneck                                                           #
        # ------------------------------------------------------------------ #
        self.res_mid1 = ResBlockSpatial(D * 4, cond_dim, context_dim, num_heads=num_heads)
        self.res_mid2 = ResBlockSpatial(D * 4, cond_dim, context_dim, num_heads=num_heads)

        # ------------------------------------------------------------------ #
        # U-Net decoder                                                        #
        # ------------------------------------------------------------------ #
        self.up1    = nn.ConvTranspose2d(D * 4, D * 2, kernel_size=4, stride=2, padding=1)
        self.res_u1 = ResBlockSpatial(D * 2, cond_dim, context_dim, num_heads=num_heads)

        self.up2    = nn.ConvTranspose2d(D * 2, D,     kernel_size=4, stride=2, padding=1)
        self.res_u2 = ResBlockSpatial(D,     cond_dim, context_dim, num_heads=num_heads)

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
        B, T_fcast, C, Hl, Wl = x.shape
        T_ctx = context.size(1)

        # Build timestep conditioning vector: (B, 4D)
        cond = self.cond_expand(self.time_emb(t))

        # Flatten forecast frames into batch dim for spatial processing
        x_flat   = x.view(B * T_fcast, C, Hl, Wl)

        # Expand cond and context to match flattened batch (B*T_fcast)
        cond_exp = cond.unsqueeze(1).expand(-1, T_fcast, -1).reshape(B * T_fcast, -1)
        ctx_exp  = (
            context
            .unsqueeze(1)                              # (B, 1, T_ctx, D)
            .expand(-1, T_fcast, -1, -1)               # (B, T_fcast, T_ctx, D)
            .reshape(B * T_fcast, T_ctx, -1)           # (B*T_fcast, T_ctx, D)
        )

        # U-Net forward
        h0 = self.entry(x_flat)

        h1 = self.down1(h0)
        h1 = self.res_d1(h1, cond_exp, ctx_exp)

        h2 = self.down2(h1)
        h2 = self.res_d2(h2, cond_exp, ctx_exp)

        h  = self.res_mid1(h2, cond_exp, ctx_exp)
        h  = self.res_mid2(h,  cond_exp, ctx_exp)

        h  = self.up1(h) + h1
        h  = self.res_u1(h, cond_exp, ctx_exp)

        h  = self.up2(h) + h0
        h  = self.res_u2(h, cond_exp, ctx_exp)

        return self.exit(h).view(B, T_fcast, C, Hl, Wl)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class LatentDiffusionTransformer(nn.Module):
    """
    Latent Diffusion Transformer (LDT) for probabilistic solar irradiance
    nowcasting from satellite image sequences.

    Components
    ----------
    VAE
        Compresses (C, H, W) → (latent_channels, H/8, W/8).
        Frozen during diffusion training (Stage 2).

    SpatialFrameEncoder
        Per-frame CNN stem: (latent_channels, Hl, Wl) → (transformer_dim,).
        Replaces the flat linear projection; preserves local spatial structure.
        Applied separately to the irradiance channel and the BARRA channels.

    TransformerBackbone
        Temporally encodes the irradiance token sequence.
        Conditioned on the diffusion timestep via AdaLN.
        Receives BARRA tokens (solar elevation + atmospheric instability) as
        auxiliary context via cross-attention (context_dim=barra_token_dim).

    DenoiserNetwork
        Spatial U-Net predicting noise ε in the diffusion forward process.
        Every ResBlock is conditioned on:
          - Diffusion timestep via FiLM.
          - Full transformer context sequence via learned cross-attention
            (replaces mean-pooling from the previous version).

    Channel split
    -------------
    Input images have 3 channels:
        [0]   irradiance  (satellite)        → encoded separately → transformer
        [1:3] BARRA channels (solar elev. +    → encoded separately → transformer
              atmos. instability)              cross-attention context
    """

    # Eval noise levels for forward_eval — fixed set for reproducible metrics
    EVAL_TIMESTEPS = [0, 249, 499, 749, 999]

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
        barra_token_dim:        int   = 128,
        num_diffusion_steps:    int   = 1000,
        denoiser_hidden_dim:    int   = 128,
        denoiser_heads:         int   = 4,
        dropout:                float = 0.1,
    ):
        """
        Args:
            image_channels:         Total satellite image channels (3: irr, elev, instab).
            image_size:             Spatial size of input images (must be divisible by 8).
            latent_channels:        VAE latent map channels (4 is standard).
            vae_hidden_dim:         Base channel width of the VAE conv stack.
            num_transformer_layers: Depth of the temporal transformer.
            num_heads:              Attention heads in the transformer.
            feedforward_dim:        FFN width inside each transformer layer.
            transformer_dim:        Token dimension of the irradiance transformer stream.
            barra_token_dim:        Token dimension for the BARRA auxiliary context stream.
                                    Passed as context_dim to TransformerBackbone so the
                                    BARRA tokens are injected via cross-attention.
            num_diffusion_steps:    Total diffusion timesteps T.
            denoiser_hidden_dim:    Base channel width of the U-Net denoiser.
            denoiser_heads:         Attention heads in denoiser cross-attention blocks.
            dropout:                Dropout rate in the transformer.
        """
        super().__init__()

        self.image_channels      = image_channels
        self.latent_channels     = latent_channels
        self.latent_size         = image_size // 8
        self.num_diffusion_steps = num_diffusion_steps
        self.transformer_dim     = transformer_dim
        self.barra_token_dim     = barra_token_dim

        # Number of BARRA channels (everything except irradiance channel 0)
        self._barra_channels = image_channels - 1   # NEED TO UPDATE WHEN CHANNELS ARE CHANGED

        # ------------------------------------------------------------------ #
        # VAE                                                                #
        # ------------------------------------------------------------------ #
        self.vae = VariationalAutoencoder(
            image_channels  = image_channels,
            latent_channels = latent_channels,
            hidden_dim      = vae_hidden_dim,
            image_size      = image_size,
        )

        # ------------------------------------------------------------------ #
        # Spatial frame encoders (CNN stem, one per stream)                   #
        # ------------------------------------------------------------------ #

        # Irradiance stream: 1 input channel (channel 0 of the latent)
        # The VAE encodes all channels jointly → latent_channels output channels.
        # We split the latent along the channel axis: irradiance gets all
        # latent_channels (the VAE mixes them), BARRA gets its own encoder.
        # In practice the simplest correct split is to encode the full latent
        # for irradiance and encode the raw BARRA images (pre-VAE) for context,
        # since BARRA data is low-frequency and doesn't need VAE compression.
        self.irr_frame_encoder = SpatialFrameEncoder(
            latent_channels = latent_channels,
            token_dim       = transformer_dim,
        )

        # BARRA stream: encodes raw BARRA image channels (solar elev + instability)
        # directly in pixel space — no VAE needed for smooth scalar fields.
        self.barra_frame_encoder = SpatialFrameEncoder(
            latent_channels = self._barra_channels,
            token_dim       = barra_token_dim,
        )

        # ------------------------------------------------------------------ #
        # Transformer backbone                                                 #
        # ------------------------------------------------------------------ #
        self.transformer = TransformerBackbone(
            latent_dim      = transformer_dim,
            num_layers      = num_transformer_layers,
            num_heads       = num_heads,
            feedforward_dim = feedforward_dim,
            dropout         = dropout,
            mask_mode       = "bidirectional",
            context_dim     = barra_token_dim,   # wires BARRA into cross-attention
        )

        # ------------------------------------------------------------------ #
        # Denoiser                                                             #
        # ------------------------------------------------------------------ #
        self.denoiser = DenoiserNetwork(
            latent_channels = latent_channels,
            latent_size     = self.latent_size,
            context_dim     = transformer_dim,
            hidden_dim      = denoiser_hidden_dim,
            num_heads       = denoiser_heads,
        )

        # ------------------------------------------------------------------ #
        # Cosine noise schedule (Nichol & Dhariwal 2021)                      #
        # ------------------------------------------------------------------ #
        betas               = self._cosine_beta_schedule(num_diffusion_steps)
        alphas              = 1.0 - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
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
        images:        torch.Tensor,   # (B, T, C, H, W) — full C-channel images
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Encode a full C-channel frame sequence to spatial latent maps via the VAE.
    
        The VAE was trained on all image_channels jointly, so the full image
        must always be passed here — never a single-channel slice.
        The caller is responsible for splitting channels before/after as needed.
    
        Returns:
            latents: (B, T, latent_channels, Hl, Wl)
        """
        assert images.ndim == 5, f"Expected (B, T, C, H, W), got {images.shape}"
        assert images.shape[2] == self.image_channels, (
            f"VAE expects {self.image_channels} channels, got {images.shape[2]}. "
            f"Always pass the full image to encode_images."
        )
        B, T = images.shape[:2]
        flat = images.view(B * T, *images.shape[2:])
        mu, logvar = self.vae.encode(flat)
        z = mu if deterministic else self.vae.reparameterize(mu, logvar)
        return z.view(B, T, self.latent_channels, z.shape[-2], z.shape[-1])


    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode spatial latent maps to image frames.

        Args:
            latents: (B, T, latent_channels, Hl, Wl)
        Returns:
            images:  (B, T, C, H, W)
        """
        assert latents.ndim == 5, f"Expected (B, T, LC, Hl, Wl), got {latents.shape}"
        B, T = latents.shape[:2]
        flat    = latents.view(B * T, *latents.shape[2:])
        decoded = self.vae.decode(flat)
        return decoded.view(B, T, *decoded.shape[1:])

    def _encode_context(
        self,
        context_latents: torch.Tensor,   # (B, T, latent_channels, Hl, Wl)
        barra_images:      torch.Tensor,   # (B, T, barra_channels, H, W)
        timesteps:       torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """
        Encode the context sequence for use as denoiser conditioning.

        Steps:
          1. CNN-encode each irradiance latent frame → irradiance tokens (B, T, D).
          2. CNN-encode each BARRA image frame         → BARRA tokens (B, T, barra_dim).
          3. Pass irradiance tokens through the transformer, with:
               - AdaLN conditioning on the diffusion timestep.
               - Cross-attention conditioning on the BARRA token sequence.

        Args:
            context_latents: VAE latents for the irradiance channel.
            barra_images:      Raw BARRA frames (solar elevation + instability).
            timesteps:       Diffusion timestep indices, shape (B,).

        Returns:
            encoded: (B, T, transformer_dim)
        """
        B, T = context_latents.shape[:2]

        # 1. Encode irradiance latent frames with CNN stem
        irr_flat    = context_latents.view(B * T, *context_latents.shape[2:])
        irr_tokens  = self.irr_frame_encoder(irr_flat)          # (B*T, D)
        irr_tokens  = irr_tokens.view(B, T, -1)                 # (B, T, D)

        # 2. Encode BARRA frames with CNN stem
        barra_flat    = barra_images.view(B * T, *barra_images.shape[2:])
        barra_tokens  = self.barra_frame_encoder(barra_flat)          # (B*T, barra_dim)
        barra_tokens  = barra_tokens.view(B, T, -1)                 # (B, T, barra_dim)

        # 3. Temporally encode irradiance tokens, cross-attending to barra tokens
        return self.transformer(
            irr_tokens,
            timesteps = timesteps,
            context   = barra_tokens,
        )

    # ------------------------------------------------------------------ #
    # Forward diffusion                                                    #
    # ------------------------------------------------------------------ #

    def add_noise(
        self,
        x:         torch.Tensor,   # (B, T, latent_channels, Hl, Wl)
        timesteps: torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise       = torch.randn_like(x)
        sqrt_ab     = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        sqrt_one_ab = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        return sqrt_ab * x + sqrt_one_ab * noise, noise

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
    
        The full C-channel image is passed to the VAE for encoding.
        BARRA channels (indices 1:) are also extracted in pixel space and
        injected into the transformer via cross-attention, independently
        of the VAE latent stream.
        """
        B      = context_images.shape[0]
        device = context_images.device
    
        # BARRA channels extracted in pixel space for the transformer cross-attention
        ctx_barra = context_images[:, :, 1:, :, :]   # (B, T, 2, H, W)
    
        # Full image passed to VAE — it was trained on all channels jointly
        context_latents = self.encode_images(context_images, deterministic=True)
        target_latents  = self.encode_images(target_images,  deterministic=True)
    
        t = torch.randint(0, self.num_diffusion_steps, (B,), device=device)
    
        encoded_context = self._encode_context(context_latents, ctx_barra, t)
    
        noisy_targets, noise = self.add_noise(target_latents, t)
        predicted_noise      = self.denoiser(noisy_targets, t, encoded_context)
    
        loss = F.mse_loss(predicted_noise, noise)
        return loss, encoded_context

    # ------------------------------------------------------------------ #
    # Evaluation forward pass                                              #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def forward_eval(
        self,
        context_images: torch.Tensor,
        target_images:  torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B      = context_images.shape[0]
        device = context_images.device
    
        ctx_barra    = context_images[:, :, 1:, :, :]
        tgt_irr    = target_images[:,  :, 0:1, :, :]   # irradiance only for metrics
    
        context_latents = self.encode_images(context_images, deterministic=True)
        target_latents  = self.encode_images(target_images,  deterministic=True)
    
        latent_losses, irr_mses, irr_maes = [], [], []
    
        for t_val in self.EVAL_TIMESTEPS:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
    
            encoded_context      = self._encode_context(context_latents, ctx_barra, t)
            noisy_targets, noise = self.add_noise(target_latents, t)
            predicted_noise      = self.denoiser(noisy_targets, t, encoded_context)
    
            latent_losses.append(F.mse_loss(predicted_noise, noise))
    
            sqrt_ab     = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1, 1)
            sqrt_one_ab = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1, 1)
            pred_x0     = (noisy_targets - sqrt_one_ab * predicted_noise) / sqrt_ab.clamp(min=1e-8)
    
            # Decode full latent → full image, then slice irradiance channel
            pred_images = self.decode_latents(pred_x0)          # (B, T, C, H, W)
            pred_irr    = pred_images[:, :, 0:1, :, :]
    
            irr_mses.append(F.mse_loss(pred_irr, tgt_irr))
            irr_maes.append(F.l1_loss(pred_irr,  tgt_irr))
    
        return {
            "latent_loss":    torch.stack(latent_losses).mean(),
            "irradiance_mse": torch.stack(irr_mses).mean(),
            "irradiance_mae": torch.stack(irr_maes).mean(),
        }

    # ------------------------------------------------------------------ #
    # Inference: DDIM reverse diffusion                                    #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def sample(
        self,
        context_images:     torch.Tensor,   # (B, T_ctx, C, H, W) — full C channels
        num_forecast_steps: int,
        num_samples:        int   = 1,
        ddim_steps:         int   = 50,
        eta:                float = 0.0,
    ) -> torch.Tensor:
        B      = context_images.shape[0]
        device = context_images.device
    
        if num_samples > 1:
            context_images = context_images.repeat_interleave(num_samples, dim=0)
            B = context_images.shape[0]
    
        ctx_barra = context_images[:, :, 1:, :, :]
    
        # Full image to VAE
        context_latents = self.encode_images(context_images, deterministic=True)
    
        step_ratio     = max(self.num_diffusion_steps // ddim_steps, 1)
        ddim_timesteps = list(
            reversed(range(0, self.num_diffusion_steps, step_ratio))
        )[:ddim_steps]
    
        x = torch.randn(
            B, num_forecast_steps,
            self.latent_channels, self.latent_size, self.latent_size,
            device=device,
        )
    
        for i, t_val in enumerate(ddim_timesteps):
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
    
            encoded_context = self._encode_context(context_latents, ctx_barra, t_tensor)
            pred_noise      = self.denoiser(x, t_tensor, encoded_context)
    
            alpha_bar      = self.alphas_cumprod[t_val]
            t_prev         = ddim_timesteps[i + 1] if i + 1 < len(ddim_timesteps) else -1
            alpha_bar_prev = (
                self.alphas_cumprod[t_prev]
                if t_prev >= 0
                else torch.tensor(1.0, device=device)
            )
    
            sqrt_ab   = torch.sqrt(alpha_bar).clamp(min=1e-8)
            pred_x0   = (x - torch.sqrt(1.0 - alpha_bar) * pred_noise) / sqrt_ab
    
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
