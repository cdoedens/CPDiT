"""
Spatial VAE for satellite image latent space compression.

Encoder: (C, H, W) → (latent_channels, H/8, W/8)
Decoder: (latent_channels, H/8, W/8) → (C, H, W)

For H=W=256 this gives a 32×32 latent map — spatially structured,
not a flat vector. The diffusion model then operates on this map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim
from pytorch_msssim import ms_ssim


class ResBlock(nn.Module):
    """Conv residual block with GroupNorm + SiLU."""

    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DownBlock(nn.Module):
    """Strided conv downsample + residual refinement."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1
        )
        self.res = ResBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class UpBlock(nn.Module):
    """Transposed conv upsample + residual refinement."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1
        )
        self.res = ResBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.up(x))


class VariationalAutoencoder(nn.Module):
    """
    Spatial convolutional VAE.

    Produces a (latent_channels, H/8, W/8) latent map rather than a flat
    vector. This preserves spatial structure and gives the downstream
    diffusion model a meaningful 2-D latent space to operate on.

    For image_size=256: latent shape is (latent_channels, 32, 32).
    """

    def __init__(
        self,
        image_channels:  int = 3,
        latent_channels: int = 4,
        hidden_dim:      int = 128,
        image_size:      int = 256,
    ):
        """
        Args:
            image_channels:  Input/output image channels.
            latent_channels: Number of channels in the latent map.
                             4 is standard (matches Stable Diffusion).
            hidden_dim:      Base channel width. Progression: D → 2D → 4D.
            image_size:      Spatial size of square input (must be div by 8).
        """
        super().__init__()

        assert image_size % 8 == 0, (
            f"image_size must be divisible by 8, got {image_size}"
        )

        self.image_channels  = image_channels
        self.latent_channels = latent_channels
        self.hidden_dim      = hidden_dim
        self.image_size      = image_size
        self.latent_size     = image_size // 8  # spatial size of latent map

        D = hidden_dim

        # ------------------------------------------------------------------ #
        # Encoder: (C, H, W) → (4D, H/8, W/8)                               #
        # ------------------------------------------------------------------ #
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, D, kernel_size=3, padding=1),  # entry
            ResBlock(D),
            DownBlock(D,     D * 2),   # H/2
            ResBlock(D * 2),
            DownBlock(D * 2, D * 4),   # H/4
            ResBlock(D * 4),
            DownBlock(D * 4, D * 4),   # H/8
            ResBlock(D * 4),
            nn.GroupNorm(8, D * 4),
            nn.SiLU(),
        )

        # Project to mu and logvar — both spatial maps
        self.conv_mu     = nn.Conv2d(D * 4, latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv2d(D * 4, latent_channels, kernel_size=1)

        # ------------------------------------------------------------------ #
        # Decoder: (latent_channels, H/8, W/8) → (C, H, W)                  #
        # ------------------------------------------------------------------ #
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, D * 4, kernel_size=1),  # entry
            ResBlock(D * 4),
            UpBlock(D * 4, D * 4),     # H/4
            ResBlock(D * 4),
            UpBlock(D * 4, D * 2),     # H/2
            ResBlock(D * 2),
            UpBlock(D * 2, D),         # H
            ResBlock(D),
            nn.GroupNorm(8, D),
            nn.SiLU(),
            nn.Conv2d(D, image_channels, kernel_size=3, padding=1),  # exit
        )

    # ------------------------------------------------------------------ #
    # Encode                                                               #
    # ------------------------------------------------------------------ #

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            mu:     (B, latent_channels, H/8, W/8)
            logvar: (B, latent_channels, H/8, W/8)
        """
        h      = self.encoder(x)
        mu     = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar

    # ------------------------------------------------------------------ #
    # Reparameterise                                                        #
    # ------------------------------------------------------------------ #

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std    = torch.exp(0.5 * logvar)
        eps    = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------ #
    # Decode                                                               #
    # ------------------------------------------------------------------ #

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_channels, H/8, W/8)
        Returns:
            x_recon: (B, C, H, W)
        """
        return self.decoder(z)

    def encode_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_recon    = self.decode(z)
        return x_recon, mu, logvar

    # ------------------------------------------------------------------ #
    # Loss                                                                 #
    # ------------------------------------------------------------------ #

    def vae_loss(
        self,
        x:       torch.Tensor,
        x_recon: torch.Tensor,
        mu:      torch.Tensor,
        logvar:  torch.Tensor,
        beta:    float = 0.001,
        ssim_weight: float = 0.4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruction (MSE + SSIM) + beta-weighted KL divergence.

        KL is computed over the full spatial latent map and normalised
        by the number of latent elements so it stays on the same scale
        as the reconstruction loss regardless of latent_channels or
        image_size.
        """
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        ssim_loss  = 1.0 - ssim(x_recon, x, data_range=1.0)

        # KL over spatial map, mean-reduced to match recon scale
        kl_loss = -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        ).mean()

        total = recon_loss + ssim_weight * ssim_loss + beta * kl_loss
        return total, recon_loss, kl_loss

    def extra_repr(self) -> str:
        return (
            f"image_channels={self.image_channels}, "
            f"image_size={self.image_size}, "
            f"latent_channels={self.latent_channels}, "
            f"latent_size={self.latent_size}, "
            f"hidden_dim={self.hidden_dim}"
        )
