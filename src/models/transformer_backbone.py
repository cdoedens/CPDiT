"""Transformer backbone for temporal modeling."""

import math

import torch
import torch.nn as nn


class TransformerBackbone(nn.Module):
    """Transformer-based temporal encoder for satellite image sequences."""

    def __init__(self, latent_dim=256, num_layers=4, num_heads=8, feedforward_dim=1024, dropout=0.1, max_seq_len=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len

        self.register_buffer("positional_encoding", self._create_positional_encoding(max_seq_len, latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.input_projection = nn.Linear(latent_dim, latent_dim)
        self.output_projection = nn.Linear(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, latent_seq, timestamps=None, padding_mask=None):
        seq_len = latent_seq.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}.")

        if timestamps is not None:
            pe = self.time_mlp(timestamps.float())
        else:
            pe = self.positional_encoding[:, :seq_len, :].to(latent_seq.device)

        x = self.input_projection(latent_seq)
        x = self.dropout(x + pe)
        causal_mask = self._create_causal_mask(seq_len, x.device)
        encoded = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        encoded = self.norm(encoded)
        return self.output_projection(encoded)

    def extra_repr(self):
        return f"latent_dim={self.latent_dim}, max_seq_len={self.max_seq_len}"


