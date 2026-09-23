"""
The two learned networks: the temporal context encoder and the DiT score network.

Everything in this file has parameters. The diffusion mathematics that drives
them lives in `diffusion.py`, and the assembly that wires them together lives in
`latent_diffusion.py`.

Where this sits in the workflow
-------------------------------
    vae.py encodes each frame to a (C, H/8, W/8) latent map, then:

      context latents (B, T_ctx, C, Hl, Wl)
             |
             |  flatten + linear  (done in latent_diffusion.py)
             v
      +----------------------+
      | ContextEncoder       |  section 2 — "what has the sky been doing?"
      | temporal transformer |  Summarises the recent past into T_ctx tokens.
      +----------------------+
             |
             |  pooled into the conditioning vector
             v
      +----------------------------------------------+
      | DiTDenoiser                                  |  section 3 — the score
      | patch embed -> N x (neighbourhood attn +     |  network the sampler
      | MLP, adaLN-Zero) -> unpatchify               |  calls at every step
      +----------------------------------------------+
             |
             v
      sigma-scaled score residual, which
      LatentDiffusionTransformer.score() turns into grad_x log p_t(x)

The context reaches the denoiser by two different routes, deliberately:

  - *Spatially*, by concatenating the context latent frames to the noisy latent
    along the channel axis before patch embedding, so the denoiser keeps the
    full spatial structure of the recent past. A pooled vector cannot say
    *where* the clouds are.
  - *Globally*, through adaLN-Zero, from the pooled ContextEncoder summary
    together with the diffusion time and the forecast lead time.

References:
  - Peebles & Xie 2023   — https://arxiv.org/abs/2212.09748  (DiT, adaLN-Zero)
  - Hassani et al. 2023  — https://arxiv.org/abs/2204.07143  (Neighbourhood Attention)
  - Vaswani et al. 2017  — https://arxiv.org/abs/1706.03762  (Transformer)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Shared building blocks
#
#    Used by both networks below, so they live here rather than being defined
#    multiple times.
# =============================================================================

def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """
    Standard 2-D sine-cosine positional embedding (as used by ViT / DiT / MAE).

    PURPOSE: Positional embeddings are added to the patch embeddings to give the model
             a sense of spatial location, because transformers do not inherently understand
             spatial relationships. The embedding is constructed by applying sine and cosine
             functions of different frequencies to the grid coordinates.

    Returns:
        (1, grid_h * grid_w, embed_dim)
    """
    if embed_dim % 4 != 0:
        raise ValueError(
            f"embed_dim must be divisible by 4 for 2-D sin-cos embedding, got {embed_dim}"
        )

    def _1d(dim: int, pos: torch.Tensor) -> torch.Tensor:
        omega = torch.arange(dim // 2, dtype=torch.float64)
        omega = 1.0 / (10000.0 ** (omega / (dim / 2.0)))
        out = pos.reshape(-1).double()[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    # grid[0] varies along w (x axis), grid[1] varies along h (y axis)
    gh = torch.arange(grid_h, dtype=torch.float64)
    gw = torch.arange(grid_w, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(gh, gw, indexing="ij")

    emb_x = _1d(embed_dim // 2, grid_x)
    emb_y = _1d(embed_dim // 2, grid_y)
    emb = torch.cat([emb_y, emb_x], dim=1)          # (grid_h*grid_w, embed_dim)
    return emb.float().unsqueeze(0)


class TimestepEmbedder(nn.Module):
    """
    Sinusoidal diffusion-timestep embedding followed by a two-layer MLP.

    PURPOSE: informs the denoiser about the current diffusion step, allowing it to adapt
             its predictions based on how much noise is present in the input.

    Uses the standard DDPM/DiT frequency scaling (``exp(-log(10000) * i / half)``).
    """

    def __init__(self, hidden_dim: int, frequency_embedding_dim: int = 256):
        super().__init__()
        if frequency_embedding_dim % 2 != 0:
            raise ValueError("frequency_embedding_dim must be even.")
        self.frequency_embedding_dim = frequency_embedding_dim
        # The MLP is a simple feedforward network that processes the sinusoidal embeddings to produce a richer representation of the timestep.
        # Converts shape (B, frequency_embedding_dim) -> (B, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """(B,) -> (B, hidden_dim)"""
        return self.mlp(
            self.sinusoidal_embedding(timesteps, self.frequency_embedding_dim)
        )

# ---------------------------------------------------------------------------
# Optional NATTEN fast path
# ---------------------------------------------------------------------------
# NATTEN provides fused CUDA kernels for neighbourhood attention that are far
# cheaper than the masked-softmax fallback on large token grids. The installed
# wheel is built against a specific torch version, so importing it can fail on
# a mismatched environment. We probe once, at import time, and silently fall
# back to the exact PyTorch implementation if anything goes wrong.

# NATTEN provides fused CUDA kernels for neighbourhood attention that are far
# cheaper than the masked-softmax fallback on large token grids. The installed
# wheel is built against a specific torch version, so importing it can fail on
# a mismatched environment. We probe once, at import time, and silently fall
# back to the exact PyTorch implementation if anything goes wrong.

try:  # pragma: no cover - depends on the deployment environment
    from natten.functional import na2d as _natten_na2d

    _NATTEN_AVAILABLE = True
except Exception as _exc:  # noqa: BLE001 - any failure means "no fast path"
    _natten_na2d = None
    _NATTEN_AVAILABLE = False
    logger.info(
        "NATTEN unavailable (%s: %s) — using the exact PyTorch neighbourhood "
        "attention fallback.",
        type(_exc).__name__,
        _exc,
    )


# =============================================================================
# 2. ContextEncoder — the temporal half of the model
#
#    Takes the sequence of context latent tokens and lets each frame attend to
#    every other, producing a temporally-aware summary of the recent past. This
#    is what carries "the cloud field has been moving north-east for the last
#    hour" into the denoiser.
#
#    Note this network is NOT conditioned on the diffusion time. It only ever
#    sees clean context latents, so there is no noise level to tell it about.
#    (An earlier version fed it a diffusion timestep through adaLN — but the
#    caller always passed t=0, so 446k of its parameters, 25% of the module,
#    were computing a constant. That path is gone; the per-block affine and
#    gates below express the same thing with ~2k parameters.)
# =============================================================================


class ContextEncoderBlock(nn.Module):
    """
    Standard pre-norm transformer block with zero-initialised residual gates.

    This is a typical transformer block:
    - layer norm, multi-head attention,
    - residual connection, layer norm,
    - feedforward MLP,
    - residual connection.
    The only twist is that the two residuals are gated by learnable scalars that start at zero.

    The gates start at zero so the block begins as the identity, which is the
    stabilising property the previous adaLN-Zero formulation provided.
    """

    def __init__(self, dim: int, num_heads: int, feedforward_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, dim),
            nn.Dropout(dropout),
        )
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_ffn = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.gate_attn.tanh() * attn_out
        x = x + self.gate_ffn.tanh() * self.ffn(self.norm2(x))
        return x


class ContextEncoder(nn.Module):
    """
    Temporal encoder over the context latent sequence, using ContextEncoderBlock.

    PURPOSE: Summarises the recent past into a temporally-aware representation
             that the denoiser can use to condition its predictions.


    Each context frame is represented by a latent token, and the transformer
    lets each token attend to every other, so the time varying patterns of the
    recent past can be captured.

    Input:  (B, T_ctx, latent_dim) tokens, one per context frame
    Output: (B, T_ctx, latent_dim) temporally-encoded tokens
    """

    def __init__(
        self,
        latent_dim:      int   = 512,
        num_layers:      int   = 4,
        num_heads:       int   = 8,
        feedforward_dim: int   = 1024,
        dropout:         float = 0.1,
        max_seq_len:     int   = 64,
    ):
        super().__init__()
        if latent_dim % 2 != 0:
            raise ValueError(f"latent_dim must be even, got {latent_dim}.")

        self.latent_dim  = latent_dim
        self.num_layers  = num_layers
        self.max_seq_len = max_seq_len

        # Frames arrive as an unordered set as far as attention is concerned,
        # so sequence position has to be supplied explicitly.
        self.register_buffer(
            "positional_encoding",
            self._sinusoidal_positions(max_seq_len, latent_dim),
            persistent=False,
        )

        self.blocks = nn.ModuleList([
            ContextEncoderBlock(latent_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers) # repeat ContextEncoderBlock num_layers times to build the full ContextEncoder
        ])

        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_projection = nn.Linear(latent_dim, latent_dim)

    @staticmethod
    def _sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
        """Classic sine/cosine positional encoding, shape (1, max_len, d_model)."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, latent_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_seq: (B, T, latent_dim) — one token per context frame.
        Returns:
            (B, T, latent_dim)
        """
        _, T, _ = latent_seq.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}."
            )

        x = latent_seq + self.positional_encoding[:, :T, :].to(latent_seq.dtype)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x) # pass the input through each ContextEncoderBlock in the sequence

        return self.output_projection(self.output_norm(x))

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.latent_dim}, num_layers={self.num_layers}, "
            f"max_seq_len={self.max_seq_len}"
        )


# =============================================================================
# 3. DiTDenoiser — the score network
#
#    Called once per solver step during sampling, so this is where essentially
#    all inference compute goes. Given a perturbed latent and the diffusion
#    time, it emits the sigma-scaled residual that becomes the score.
# =============================================================================

def neighbourhood_mask(grid_h: int, grid_w: int, window: int) -> torch.Tensor:
    """
    Boolean self-attention mask implementing NATTEN neighbourhood semantics.

    Every query attends to a ``window x window`` block of keys centred on
    itself. Near a boundary the window is *shifted inward* rather than
    truncated, so each query attends to exactly ``window**2`` keys — this is
    what distinguishes neighbourhood attention from a naive sliding window.

    Returns:
        (grid_h * grid_w, grid_h * grid_w) bool tensor, True where attention
        is allowed.
    """
    radius = window // 2

    def _axis(size: int) -> torch.Tensor:
        idx = torch.arange(size)
        # Clamp the window centre so the full window stays inside the grid.
        centre = idx.clamp(min=radius, max=max(size - 1 - radius, radius))
        return (idx.unsqueeze(0) - centre.unsqueeze(1)).abs() <= radius

    row_ok = _axis(grid_h)                                  # (H, H)
    col_ok = _axis(grid_w)                                  # (W, W)

    mask = row_ok[:, None, :, None] & col_ok[None, :, None, :]
    return mask.reshape(grid_h * grid_w, grid_h * grid_w)


class NeighbourhoodAttention2D(nn.Module):
    """
    Multi-head 2-D neighbourhood attention over a token grid.

    Tokens arrive flattened as ``(B, H*W, D)`` in row-major order. The fused
    NATTEN kernel is used when available; otherwise an exact masked-softmax
    implementation is used, which produces identical results at O(N^2) cost.
    The fallback is entirely adequate for the token grids this model uses
    (a 256px image gives a 32x32 latent and an 8x8 token grid at patch size 4).
    """

    def __init__(
        self,
        dim:         int,
        num_heads:   int,
        window_size: int,
        qkv_bias:    bool = True,
        use_natten:  Optional[bool] = None,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}.")

        self.dim         = dim
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.scale       = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias) # qkv = query, key, value
        self.proj = nn.Linear(dim, dim)

        self.use_natten = _NATTEN_AVAILABLE if use_natten is None else use_natten
        if self.use_natten and not _NATTEN_AVAILABLE:
            raise RuntimeError("use_natten=True but NATTEN could not be imported.")

        # Masks are small and depend only on (grid_h, grid_w, effective window),
        # so cache them per shape instead of rebuilding every forward pass.
        self._mask_cache: dict[tuple[int, int, int], torch.Tensor] = {}

    def effective_window(self, grid_h: int, grid_w: int) -> int:
        """
        Largest odd window that fits inside the token grid.

        A window wider than the grid is meaningless — neighbourhood attention
        degenerates to global attention — so we clamp rather than error.
        """
        limit = min(grid_h, grid_w)
        window = min(self.window_size, limit)
        if window % 2 == 0:                 # neighbourhood windows must be odd
            window -= 1
        return max(window, 1)

    def _get_mask(self, grid_h: int, grid_w: int, window: int, device) -> torch.Tensor:
        key = (grid_h, grid_w, window)
        mask = self._mask_cache.get(key)
        if mask is None or mask.device != device:
            mask = neighbourhood_mask(grid_h, grid_w, window).to(device)
            self._mask_cache[key] = mask
        return mask

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """
        Args:
            x:      (B, grid_h * grid_w, dim)
            grid_h: token grid height
            grid_w: token grid width
        Returns:
            (B, grid_h * grid_w, dim)
        """
        B, L, D = x.shape
        if L != grid_h * grid_w:
            raise ValueError(f"Token count {L} != grid {grid_h}x{grid_w}.")

        window = self.effective_window(grid_h, grid_w)

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim) # qkv = query, key, value

        if self.use_natten: # use NATTEN module if available
            # NATTEN expects (B, H, W, heads, head_dim).
            q, k, v = (
                qkv[:, :, i].reshape(B, grid_h, grid_w, self.num_heads, self.head_dim)
                for i in range(3)
            )
            out = _natten_na2d(q, k, v, kernel_size=window, dilation=1, scale=self.scale)
            out = out.reshape(B, L, D)
        else: # scaled dot product attention with neighbourhood mask
            # (B, heads, L, head_dim)
            q, k, v = (qkv[:, :, i].permute(0, 2, 1, 3) for i in range(3))
            mask = self._get_mask(grid_h, grid_w, window, x.device)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask.unsqueeze(0).unsqueeze(0), scale=self.scale
            )
            out = out.permute(0, 2, 1, 3).reshape(B, L, D)

        return self.proj(out)


# ---------------------------------------------------------------------------
# DiT block
# ---------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Implement adaLN modulation
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    Standard DiT block: neighbourhood self-attention + MLP, both wrapped in
    adaptive layer norm with zero-initialised gates (adaLN-Zero).

    The conditioning vector supplies six modulation vectors per block
    (shift/scale/gate for attention and for the MLP). Because the modulation
    projection is zero-initialised, every block starts as the identity, which
    is what makes deep DiTs stable to train from scratch.
    """

    def __init__(
        self,
        dim:         int,
        num_heads:   int,
        window_size: int,
        cond_dim:    int,
        mlp_ratio:   float = 4.0,
        use_natten:  Optional[bool] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn  = NeighbourhoodAttention2D(
            dim, num_heads, window_size, use_natten=use_natten
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim, bias=True),
        )
        # adaLN-Zero: identity at initialisation.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )
        x = x + gate_a.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_a, scale_a), grid_h, grid_w
        )
        x = x + gate_m.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_m, scale_m)
        )
        return x


class FinalLayer(nn.Module):
    """adaLN-modulated projection from tokens back to latents."""

    def __init__(self, dim: int, patch_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm   = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


# ---------------------------------------------------------------------------
# DiT denoiser
# ---------------------------------------------------------------------------

class DiTDenoiser(nn.Module):
    """
    Diffusion Transformer denoiser over spatial latent maps.

    Input:  (B, T_fcast, latent_channels, Hl, Wl) perturbed forecast latents
    Output: (B, T_fcast, latent_channels, Hl, Wl) sigma-scaled score residual,
            which `LatentDiffusionTransformer.score` turns into the score.

    Conditioning has three parts:
      - the continuous diffusion time,
      - a pooled summary of the temporally-encoded context sequence,
      - the forecast lead time (which frame of the horizon this is),
    summed into one vector that drives adaLN-Zero in every block. The context
    latents themselves additionally enter as extra input channels so the
    denoiser retains full spatial detail about the recent past.
    """

    def __init__(
        self,
        latent_channels:     int,
        latent_size:         int,
        context_dim:         int,
        num_context_frames:  int,
        patch_size:          int   = 4,
        embed_dim:           int   = 768,
        depth:               int   = 16,
        num_heads:           int   = 12,
        window_size:         int   = 31,
        mlp_ratio:           float = 4.0,
        cond_dim:            int   = 256,
        max_forecast_steps:  int   = 32,
        use_natten:          Optional[bool] = None,
        time_scale:          float = 1000.0,
    ):
        super().__init__()

        if latent_size % patch_size != 0:
            raise ValueError(
                f"latent_size {latent_size} must be divisible by patch_size {patch_size}."
            )

        self.latent_channels    = latent_channels
        self.latent_size        = latent_size
        self.patch_size         = patch_size
        self.embed_dim          = embed_dim
        self.num_context_frames = num_context_frames
        self.max_forecast_steps = max_forecast_steps
        # Diffusion time arrives as a continuous value in [0, 1]. The sinusoidal
        # embedding's frequencies are tuned for the O(1000) integer range DDPM
        # used, so rescale before embedding — otherwise every t maps into the
        # first fraction of a period and the embedding cannot resolve them.
        self.time_scale = float(time_scale)

        self.grid_size = latent_size // patch_size
        num_patches    = self.grid_size ** 2

        # Noisy frame + every context frame, stacked on the channel axis.
        in_channels = latent_channels * (1 + num_context_frames)
        self.in_channels = in_channels

        # ---- Patch embedding: 4x4 strided convolution ---------------------
        # ViT style patch embedding: a convolution with kernel size = stride = patch_size, which produces a grid of tokens
        # 32 x 32 latent grid -> 8 x 8 token grid at patch size 4, with embed_dim channels per token
        self.x_embedder = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.register_buffer(
            "pos_embed",
            get_2d_sincos_pos_embed(embed_dim, self.grid_size, self.grid_size),
            persistent=False,
        )

        # ---- Conditioning -------------------------------------------------
        self.t_embedder    = TimestepEmbedder(cond_dim)
        self.ctx_proj      = nn.Linear(context_dim, cond_dim)
        self.lead_embedder = nn.Embedding(max_forecast_steps, cond_dim)
        self.cond_mlp      = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # ---- Transformer blocks -------------------------------------------
        self.blocks = nn.ModuleList([
            DiTBlock(
                embed_dim, num_heads, window_size, cond_dim,
                mlp_ratio=mlp_ratio, use_natten=use_natten,
            )
            for _ in range(depth) # repeat DiTBlock depth times to build the full DiTDenoiser
        ])

        self.final_layer = FinalLayer(embed_dim, patch_size, latent_channels, cond_dim)

        self._init_weights()

        eff = self.blocks[0].attn.effective_window(self.grid_size, self.grid_size)
        if eff < window_size:
            logger.warning(
                "Neighbourhood window %d exceeds the %dx%d token grid; clamped to %d "
                "(this makes attention effectively global). Increase image_size, or "
                "reduce patch_size / VAE downsampling, for local attention to bite.",
                window_size, self.grid_size, self.grid_size, eff,
            )
        self.effective_window_size = eff

    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        def _basic(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_basic)

        # Patch embedder: ViT-style init on the flattened kernel.
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.x_embedder.bias)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.lead_embedder.weight, std=0.02)

        # Re-zero the adaLN projections that _basic() just overwrote.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(N, L, p*p*C) -> (N, C, H, W)"""
        C, p, g = self.latent_channels, self.patch_size, self.grid_size
        x = x.reshape(x.shape[0], g, g, p, p, C)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], C, g * p, g * p)

    # ------------------------------------------------------------------ #

    def forward(
        self,
        x:               torch.Tensor,  # (B, T_fcast, C, Hl, Wl) noisy latents
        t:               torch.Tensor,  # (B,) continuous diffusion time in [0, T]
        context_tokens:  torch.Tensor,  # (B, T_ctx, context_dim) transformer output
        context_latents: torch.Tensor,  # (B, T_ctx, C, Hl, Wl) raw context latents
    ) -> torch.Tensor:
        """
        Returns:
            predicted_noise: (B, T_fcast, C, Hl, Wl)
        """
        B, T_fcast, C, Hl, Wl = x.shape
        T_ctx = context_latents.shape[1]

        if T_ctx != self.num_context_frames:
            raise ValueError(
                f"Expected {self.num_context_frames} context frames, got {T_ctx}. "
                "The denoiser's input convolution is sized for a fixed context length."
            )
        if T_fcast > self.max_forecast_steps:
            raise ValueError(
                f"Forecast length {T_fcast} exceeds max_forecast_steps={self.max_forecast_steps}."
            )

        # ---- Build per-frame input: noisy frame + all context frames ------
        ctx = context_latents.reshape(B, T_ctx * C, Hl, Wl)
        ctx = ctx.unsqueeze(1).expand(B, T_fcast, T_ctx * C, Hl, Wl)
        inp = torch.cat([x, ctx], dim=2).reshape(B * T_fcast, self.in_channels, Hl, Wl)

        # ---- Patch embed --------------------------------------------------
        tokens = self.x_embedder(inp)                       # (N, D, g, g)
        tokens = tokens.flatten(2).transpose(1, 2)          # (N, L, D)
        tokens = tokens + self.pos_embed.to(tokens.dtype)

        # ---- Conditioning: timestep + pooled context + lead time ----------
        t_emb   = self.t_embedder(t * self.time_scale)       # (B, cond)
        ctx_emb = self.ctx_proj(context_tokens.mean(dim=1))  # (B, cond)
        base    = t_emb + ctx_emb                            # (B, cond)

        lead = self.lead_embedder(
            torch.arange(T_fcast, device=x.device)
        )                                                    # (T_fcast, cond)

        cond = base.unsqueeze(1) + lead.unsqueeze(0)         # (B, T_fcast, cond)
        cond = self.cond_mlp(cond).reshape(B * T_fcast, -1)  # (N, cond)

        # ---- Transformer --------------------------------------------------
        g = self.grid_size
        for block in self.blocks:
            tokens = block(tokens, cond, g, g)

        out = self.final_layer(tokens, cond)                 # (N, L, p*p*C)
        out = self.unpatchify(out)                           # (N, C, Hl, Wl)
        return out.reshape(B, T_fcast, C, Hl, Wl)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, patch_size={self.patch_size}, "
            f"grid={self.grid_size}x{self.grid_size}, embed_dim={self.embed_dim}, "
            f"depth={len(self.blocks)}, window={self.effective_window_size}, "
            f"natten={self.blocks[0].attn.use_natten}, time_scale={self.time_scale}"
        )
