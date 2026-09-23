"""
Stage 1 of the model: compression between pixel space and latent space.

    Encoder: (C, H, W)                     -> (latent_channels, H/8, W/8)
    Decoder: (latent_channels, H/8, W/8)   -> (C, H, W)

Where this sits in the workflow
-------------------------------
Everything downstream — the ContextEncoder in `networks.py`, the DiTDenoiser,
and the whole diffusion process in `diffusion.py` — operates on these latent
maps, never on pixels. At 256x256 with 4 latent channels that is a 64x
reduction in the number of values the diffusion model has to model, which is
the entire point of a *latent* diffusion model.

The latent is a spatial map, not a flat vector: preserving 2-D structure is what
lets the denoiser use convolutional patch embedding and neighbourhood attention,
and what lets the context frames be concatenated to the noisy latent
channel-wise.

`latent_diffusion.py` calls `encode` once per training step for every frame,
`reparameterize` + `decode` for the VAE reconstruction loss, and `decode` again
to bring samples back to pixel space. The VAE is trained jointly with the
diffusion model rather than in a separate first stage.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# torchmetrics pulls in a fairly heavy dependency chain. Import it lazily so the
# model can still be constructed (and unit-tested) in a minimal environment;
# the SSIM term itself raises if it is actually asked for and unavailable.
try:
    from torchmetrics.functional import structural_similarity_index_measure as ssim
except Exception as _exc:  # noqa: BLE001
    ssim = None
    _SSIM_IMPORT_ERROR = _exc
    logger.warning("torchmetrics unavailable (%s); SSIM loss term disabled.", _exc)


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
    """
    - Performs 2x downsampling of the spatial dimensions.
    - Uses a strided convolution rather than pooling.
    - The downsampled feature map is therefore a learned representation rather than a fixed summary of the input.
    - A residual block then refines the representation with a few more convolutional layers.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Conv2d(
            in_channels, out_channels, # can increase channels as we downsample to capture more features
            kernel_size=4, stride=2, padding=1
        )
        self.res = ResBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class UpBlock(nn.Module):
    """
    - Upsamples the latent representation back to the original image size in the decoder
    - Uses a transposed convolution for learned upsampling, followed by a residual block to refine the upsampled features.
    """

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
        """
        VAE learns the mean (mu) and log-variance (logvar) of the latent distribution for each spatial location.
        After encoding to a feature map of shape (4D, H/8, W/8), perform 2d convolutions to produce mu and logvar maps of shape (latent_channels, H/8, W/8).
        """
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
        h      = self.encoder(x) # downsample spatially and increase channels to shape: (B, 4D, H/8, W/8)
        mu     = self.conv_mu(h) # reduce channels from 4D to latent_channels for mean of latent distribution
        logvar = self.conv_logvar(h) # reduce channels from 4D to latent_channels for log-variance of latent distribution
        return mu, logvar

    # ------------------------------------------------------------------ #
    # Reparameterise                                                        #
    # ------------------------------------------------------------------ #

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Use the reparameterization trick to sample from the latent distribution.
        i.e. z = mu + std * eps, where eps ~ N(0, I) and std = exp(0.5 * logvar).
        """
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std    = torch.exp(0.5 * logvar)
        eps    = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------ #
    # Decode                                                               #
    # ------------------------------------------------------------------ #

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the reparameterized latent representation back to pixel space.

        Args:
            z: (B, latent_channels, H/8, W/8)
        Returns:
            x_recon: (B, C, H, W)
        """
        return self.decoder(z)

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
        ssim_data_range: float = 8.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruction (MSE + SSIM) + beta-weighted KL divergence.

        MSE and SSIM are computed over pixel space to measure the quality of the reconstruction. KL is computed over the latent space to regularise the latent distribution.

        KL is computed over the full spatial latent map and normalised
        by the number of latent elements so it stays on the same scale
        as the reconstruction loss regardless of latent_channels or
        image_size.

        Args:
            ssim_data_range: dynamic range of the *input data*, which is
                z-scored rather than in [0, 1]. Standardised fields sit within
                roughly +/-4 sigma, so 8.0 is the sensible default. This is a
                fixed constant rather than the batch's own min-max range, so
                the loss stays comparable from batch to batch.
        """
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        if ssim_weight > 0:
            if ssim is None:
                raise RuntimeError(
                    "SSIM loss requested (ssim_weight > 0) but torchmetrics could "
                    f"not be imported: {_SSIM_IMPORT_ERROR}"
                )
            ssim_loss = 1.0 - ssim(x_recon, x, data_range=ssim_data_range)
        else:
            ssim_loss = torch.zeros((), device=x.device, dtype=recon_loss.dtype)

        # KL over spatial map, mean-reduced to match recon scale
        kl_loss = -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        ).mean()

        total = recon_loss + ssim_weight * ssim_loss + beta * kl_loss
        return total, recon_loss, kl_loss

    def extra_repr(self) -> str:
        """
        Return a string representation of the VAE's hyperparameters.
        """
        return (
            f"image_channels={self.image_channels}, "
            f"image_size={self.image_size}, "
            f"latent_channels={self.latent_channels}, "
            f"latent_size={self.latent_size}, "
            f"hidden_dim={self.hidden_dim}"
        )
