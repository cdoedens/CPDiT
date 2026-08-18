"""Transformer backbone for temporal modeling."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Timestep Embedder
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """
    Embeds diffusion timesteps using sinusoidal frequencies followed by an MLP.
    This is the standard approach from DDPM / DiT — gives the model rich,
    frequency-diverse timestep information rather than a raw scalar mapping.
    """

    def __init__(self, latent_dim: int, frequency_embedding_dim: int = 256):
        super().__init__()
        self.frequency_embedding_dim = frequency_embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    @staticmethod
    def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Creates sinusoidal embeddings for a batch of scalar timesteps.
        timesteps: (B,) integer or float tensor of diffusion steps.
        Returns: (B, dim)
        """
        assert dim % 2 == 0, "Sinusoidal embedding dim must be even."
        device = timesteps.device
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=device) / half
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """timesteps: (B,) → returns (B, latent_dim)"""
        x = self.sinusoidal_embedding(timesteps, self.frequency_embedding_dim)
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Adaptive Layer Norm (AdaLN) Modulation
# ---------------------------------------------------------------------------

class AdaLNModulation(nn.Module):
    """
    Produces per-sample scale and shift parameters for AdaLN conditioning,
    following the DiT formulation. Conditions each transformer block on the
    diffusion timestep (and optionally auxiliary context) without polluting
    the token embedding space.

    For each block we produce 6 vectors: (shift_attn, scale_attn, gate_attn,
                                          shift_ffn,  scale_ffn,  gate_ffn)
    """

    def __init__(self, latent_dim: int, num_blocks: int):
        super().__init__()
        self.num_blocks = num_blocks
        # Single linear that produces all modulation params for every block
        self.linear = nn.Linear(latent_dim, 6 * num_blocks * latent_dim, bias=True)
        # Zero-init so modulation starts as identity at the beginning of training
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, conditioning: torch.Tensor):
        """
        conditioning: (B, latent_dim) — combined timestep + context embedding.
        Returns: list of num_blocks tuples, each (shift_a, scale_a, gate_a,
                                                    shift_f, scale_f, gate_f),
                 each of shape (B, 1, latent_dim).
        """
        out = self.linear(conditioning)                          # (B, 6*num_blocks*D)
        out = out.view(out.size(0), self.num_blocks, 6, -1)      # (B, num_blocks, 6, D)
        modulations = []
        for i in range(self.num_blocks):
            params = [out[:, i, j, :].unsqueeze(1) for j in range(6)]
            modulations.append(tuple(params))
        return modulations


# ---------------------------------------------------------------------------
# AdaLN Transformer Block
# ---------------------------------------------------------------------------

class AdaLNTransformerBlock(nn.Module):
    """
    A single pre-norm transformer block with AdaLN conditioning.
    Replaces nn.TransformerEncoderLayer so that scale/shift/gate vectors
    from the diffusion timestep can modulate both the attention and FFN
    sub-layers directly — the standard DiT block design.
    """

    def __init__(self, latent_dim: int, num_heads: int, feedforward_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(latent_dim, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        shift_a: torch.Tensor,
        scale_a: torch.Tensor,
        gate_a: torch.Tensor,
        shift_f: torch.Tensor,
        scale_f: torch.Tensor,
        gate_f: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # --- Attention sub-layer with AdaLN ---
        # Modulate normalised activations: x_mod = x * (1 + scale) + shift
        x_norm = self.norm1(x) * (1.0 + scale_a) + shift_a
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + gate_a * attn_out

        # --- FFN sub-layer with AdaLN ---
        x_norm = self.norm2(x) * (1.0 + scale_f) + shift_f
        x = x + gate_f * self.ffn(x_norm)
        return x


# ---------------------------------------------------------------------------
# Cross-Attention Block (for BARRA / spatial context injection)
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    Injects auxiliary conditioning (BARRA, spatial encoder features)
    into the temporal token stream via cross-attention.

    Queries  = temporal tokens  (B, T, D)
    Keys/Values = context tokens (B, N, D)
    """

    def __init__(self, latent_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_out = nn.LayerNorm(latent_dim)
        self.gate = nn.Parameter(torch.zeros(1))  # learned residual gate, init=0

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:       (B, T, D) — temporal token sequence.
        context: (B, N, D) — auxiliary context tokens (BARRA / spatial features).
        """
        q = self.norm_q(x)
        kv = self.norm_kv(context)
        attn_out, _ = self.cross_attn(
            q, kv, kv,
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        # Gated residual: gate is learned from zero, so cross-attn activates
        # gradually rather than disrupting early training.
        return x + self.gate.tanh() * self.norm_out(attn_out)


# ---------------------------------------------------------------------------
# Main Transformer Backbone
# ---------------------------------------------------------------------------

class TransformerBackbone(nn.Module):
    """
    Transformer-based temporal encoder for satellite image sequences.

    Key design decisions vs. the original:
    - Diffusion timestep conditioning via AdaLN (not added to token embeddings).
    - Sinusoidal + MLP timestep embedder (not a raw scalar linear).
    - Separate sinusoidal positional encoding for sequence position.
    - Optional cross-attention blocks for BARRA / spatial context injection.
    - Configurable attention mask: causal, bidirectional, or prefix-causal.
    - Input projection removed (encoder latents fed directly after PE addition).
    """

    MASK_MODES = ("causal", "bidirectional", "prefix_causal")

    def __init__(
        self,
        latent_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        mask_mode: str = "bidirectional",
        context_dim: int | None = None,
        num_prefix_tokens: int = 0,
    ):
        """
        Args:
            latent_dim:        Token / model dimension.
            num_layers:        Number of transformer blocks.
            num_heads:         Attention heads.
            feedforward_dim:   FFN hidden dimension.
            dropout:           Dropout rate.
            max_seq_len:       Maximum input sequence length.
            mask_mode:         One of 'causal' | 'bidirectional' | 'prefix_causal'.
                               - 'bidirectional'  — no mask; full context (best for
                                 fixed-window nowcasting diffusion).
                               - 'causal'         — strict autoregressive mask.
                               - 'prefix_causal'  — first num_prefix_tokens attend
                                 bidirectionally; remainder attend causally.
            context_dim:       If provided, projects auxiliary context (BARRA /
                               spatial features) to latent_dim and injects it via
                               cross-attention after every self-attention block.
            num_prefix_tokens: Number of prefix tokens for 'prefix_causal' mode.
        """
        super().__init__()

        if mask_mode not in self.MASK_MODES:
            raise ValueError(f"mask_mode must be one of {self.MASK_MODES}, got '{mask_mode}'.")

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.mask_mode = mask_mode
        self.num_prefix_tokens = num_prefix_tokens

        # --- Positional encoding (sequence position, separate from timestep) ---
        self.register_buffer(
            "positional_encoding",
            self._create_positional_encoding(max_seq_len, latent_dim),
        )

        # --- Diffusion timestep embedder ---
        self.timestep_embedder = TimestepEmbedder(latent_dim)

        # --- Optional auxiliary context projection ---
        self.context_proj = (
            nn.Linear(context_dim, latent_dim) if context_dim is not None else None
        )

        # --- Transformer blocks (AdaLN) ---
        self.blocks = nn.ModuleList([
            AdaLNTransformerBlock(latent_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])

        # --- Optional cross-attention blocks (one per transformer block) ---
        self.cross_attn_blocks = (
            nn.ModuleList([
                CrossAttentionBlock(latent_dim, num_heads, dropout)
                for _ in range(num_layers)
            ])
            if context_dim is not None else None
        )

        # --- AdaLN modulation (produces params for all blocks from timestep) ---
        self.adaLN = AdaLNModulation(latent_dim, num_layers)

        # --- Output ---
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_projection = nn.Linear(latent_dim, latent_dim)

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Positional encoding
    # ------------------------------------------------------------------

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, D)

    # ------------------------------------------------------------------
    # Attention mask construction
    # ------------------------------------------------------------------

    def _create_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor | None:
        if self.mask_mode == "bidirectional":
            return None

        if self.mask_mode == "causal":
            return torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()

        # prefix_causal: prefix tokens attend to each other bidirectionally;
        # non-prefix tokens attend causally (cannot see future tokens).
        p = self.num_prefix_tokens
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        # Allow all tokens to attend to prefix tokens
        mask[:, :p] = False
        # Allow prefix tokens to attend to all other prefix tokens
        mask[:p, :p] = False
        return mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        latent_seq: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        context_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            latent_seq:           (B, T, D)  — encoded satellite latents.
            timesteps:            (B,)        — diffusion timestep indices.
            context:              (B, N, C)   — optional auxiliary context
                                               (BARRA / spatial encoder output).
            padding_mask:         (B, T)      — True where tokens are padding.
            context_padding_mask: (B, N)      — True where context is padding.

        Returns:
            (B, T, D) — temporally encoded latents conditioned on timestep.
        """
        B, T, _ = latent_seq.shape

        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}.")

        # 1. Add sinusoidal positional encoding (sequence position only)
        x = latent_seq + self.positional_encoding[:, :T, :].to(latent_seq.device)
        x = self.dropout(x)

        # 2. Embed diffusion timestep → conditioning vector (B, D)
        t_emb = self.timestep_embedder(timesteps)

        # 3. Project auxiliary context if provided
        if context is not None and self.context_proj is not None:
            context = self.context_proj(context)  # (B, N, D)

        # 4. Compute AdaLN modulation params for all blocks at once
        modulations = self.adaLN(t_emb)

        # 5. Build attention mask
        attn_mask = self._create_attention_mask(T, x.device)

        # 6. Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulations[i]

            x = block(
                x,
                shift_a=shift_a, scale_a=scale_a, gate_a=gate_a,
                shift_f=shift_f, scale_f=scale_f, gate_f=gate_f,
                attn_mask=attn_mask,
                key_padding_mask=padding_mask,
            )

            # Cross-attend to auxiliary context after each self-attention block
            if self.cross_attn_blocks is not None and context is not None:
                x = self.cross_attn_blocks[i](x, context, context_padding_mask)

        # 7. Final norm + projection
        x = self.output_norm(x)
        return self.output_projection(x)

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.latent_dim}, num_layers={self.num_layers}, "
            f"max_seq_len={self.max_seq_len}, mask_mode={self.mask_mode}"
        )
