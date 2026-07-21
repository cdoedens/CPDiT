"""
Variational Autoencoder for satellite image latent space compression.

Architecture:
  Encoder : 3× strided Conv2d (stride=2) with GroupNorm + SiLU activations.
            Spatial dims are halved at each layer: H → H/2 → H/4 → H/8.
            Output is flattened and projected to mu and logvar via Linear layers.

  Decoder : Linear projection back to the encoder's spatial shape,
            followed by 3× ConvTranspose2d (stride=2) with GroupNorm + SiLU.
            Final layer has no activation — assumes normalised input in [-1, 1]
            or [0, 1] depending on your data pipeline.

Training:
  Stage 1 — train the VAE alone using vae_loss() (reconstruction + beta * KL).
  Stage 2 — freeze the VAE and train the diffusion model (see latent_diffusion.py).

References:
  - Kingma & Welling 2013 — https://arxiv.org/abs/1312.6114  (VAE)
  - Rombach et al. 2022   — https://arxiv.org/abs/2112.10752  (LDM / frozen VAE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    """
    Convolutional VAE that compresses satellite image frames into a compact
    1-D latent vector per frame.

    The encoder output spatial size is computed automatically from image_size
    via a dummy forward pass, so the model works correctly for any input
    resolution that is divisible by 8 (e.g. 64, 128, 256, 512).
    """

    def __init__(
        self,
        image_channels: int = 5,
        latent_dim:     int = 256,
        hidden_dim:     int = 256,
        image_size:     int = 64,
    ):
        """
        Args:
            image_channels: Number of input/output image channels (e.g. 3 spectral bands).
            latent_dim:     Dimension of the VAE latent space z.
            hidden_dim:     Base channel width of the conv encoder/decoder.
                            Channel progression is hidden_dim → 2× → 4×.
            image_size:     Spatial size of square input images (H = W = image_size).
                            Must be divisible by 8 (three stride-2 convolutions).
        """
        super().__init__()

        assert image_size % 8 == 0, (
            f"image_size must be divisible by 8 (got {image_size}). "
            f"Three stride-2 convolutions reduce spatial dims by 8×."
        )

        self.image_channels = image_channels
        self.latent_dim     = latent_dim
        self.hidden_dim     = hidden_dim
        self.image_size     = image_size

        # ------------------------------------------------------------------ #
        # Encoder                                                              #
        # Conv stack: (C, H, W) → (hidden_dim*4, H/8, W/8)                   #
        # GroupNorm(8, ...) works correctly at small batch sizes and has no   #
        # train/eval discrepancy, unlike BatchNorm.                           #
        # SiLU (Swish) is standard in modern generative models.               #
        # ------------------------------------------------------------------ #

        self.encoder = nn.Sequential(
            # (C,        H,   W  ) → (D,   H/2, W/2)
            nn.Conv2d(image_channels, hidden_dim,
                      kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            # (D,   H/2, W/2) → (2D,  H/4, W/4)
            nn.Conv2d(hidden_dim, hidden_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            # (2D,  H/4, W/4) → (4D,  H/8, W/8)
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 4),
            nn.SiLU(),
        )

        # ------------------------------------------------------------------ #
        # Compute flattened encoder output size via a single dummy forward    #
        # pass. This makes the Linear layers correct for any image_size       #
        # without manual calculation.                                         #
        # ------------------------------------------------------------------ #

        with torch.no_grad():
            dummy = torch.zeros(1, image_channels, image_size, image_size)
            enc_out = self.encoder(dummy)
            self._enc_spatial_shape = enc_out.shape[1:]        # (4D, H/8, W/8)
            self._enc_flat_dim      = enc_out.numel()          # 4D * H/8 * W/8

        # ------------------------------------------------------------------ #
        # Latent projections                                                   #
        # ------------------------------------------------------------------ #

        self.fc_mu     = nn.Linear(self._enc_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._enc_flat_dim, latent_dim)

        # ------------------------------------------------------------------ #
        # Decoder                                                              #
        # Mirrors the encoder exactly so spatial dims are restored.           #
        # No final activation — the caller is responsible for normalisation.  #
        # ------------------------------------------------------------------ #

        self.fc_decode = nn.Linear(latent_dim, self._enc_flat_dim)

        self.decoder = nn.Sequential(
            # (4D, H/8, W/8) → (2D, H/4, W/4)
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            # (2D, H/4, W/4) → (D,  H/2, W/2)
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim,
                               kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            # (D,  H/2, W/2) → (C,  H,   W  )
            nn.ConvTranspose2d(hidden_dim, image_channels,
                               kernel_size=4, stride=2, padding=1),
            # No Sigmoid — avoids gradient saturation at boundaries.
            # Normalise your data to [0, 1] or [-1, 1] before training
            # and use MSE loss directly on the raw decoder output.
        )

    # ------------------------------------------------------------------ #
    # Encode                                                               #
    # ------------------------------------------------------------------ #

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of images to posterior parameters (mu, logvar).

        Args:
            x: (B, C, H, W) satellite image batch.

        Returns:
            mu:     (B, latent_dim) posterior mean.
            logvar: (B, latent_dim) posterior log-variance.
        """
        assert x.ndim == 4, \
            f"Expected 4D input (B, C, H, W), got shape {x.shape}"
        assert x.size(1) == self.image_channels, \
            f"Expected {self.image_channels} channels, got {x.size(1)}"

        h      = self.encoder(x)
        h      = h.view(h.size(0), -1)                    # flatten spatial dims
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    # ------------------------------------------------------------------ #
    # Reparameterise                                                        #
    # ------------------------------------------------------------------ #

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterisation trick: z = mu + eps * std, eps ~ N(0, I).

        logvar is clamped to [-30, 20] before exponentiation to prevent
        overflow / underflow during early training.

        Args:
            mu:     (B, latent_dim)
            logvar: (B, latent_dim)

        Returns:
            z: (B, latent_dim) sampled latent vector.
        """
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        std    = torch.exp(0.5 * logvar)
        eps    = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------ #
    # Decode                                                               #
    # ------------------------------------------------------------------ #

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent vector back to image space.

        Args:
            z: (B, latent_dim)

        Returns:
            x_recon: (B, C, H, W) reconstructed image.
        """
        assert z.ndim == 2, \
            f"Expected 2D input (B, latent_dim), got shape {z.shape}"

        h = self.fc_decode(z)
        h = h.view(h.size(0), *self._enc_spatial_shape)   # restore spatial dims
        return self.decoder(h)

    # ------------------------------------------------------------------ #
    # Convenience: deterministic encode for inference                      #
    # ------------------------------------------------------------------ #

    def encode_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the posterior mean mu without sampling.

        Use this at inference time (e.g. inside LatentDiffusionTransformer)
        when you want a fixed, noise-free latent representation.

        Args:
            x: (B, C, H, W)

        Returns:
            mu: (B, latent_dim)
        """
        mu, _ = self.encode(x)
        return mu

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass: encode → reparameterise → decode.

        Use during Stage 1 training in conjunction with vae_loss().

        Args:
            x: (B, C, H, W) input images.

        Returns:
            x_recon: (B, C, H, W) reconstructed images.
            mu:      (B, latent_dim) posterior mean.
            logvar:  (B, latent_dim) posterior log-variance.
        """
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Beta-VAE loss: reconstruction MSE + beta-weighted KL divergence.

        Both terms are summed over their respective dimensions and then
        averaged over the batch, making beta interpretable and stable
        across different image sizes and latent dimensions.

        KL divergence (closed form for diagonal Gaussian posterior vs N(0,I)):
            KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

        Args:
            x:       (B, C, H, W) original images.
            x_recon: (B, C, H, W) reconstructed images from decode().
            mu:      (B, latent_dim) posterior mean.
            logvar:  (B, latent_dim) posterior log-variance.
            beta:    Weight on the KL term. Values in [0.001, 0.1] are typical.
                     Increase to encourage a more disentangled / regular latent space.
                     Decrease if reconstructions are blurry.

        Returns:
            total_loss: recon_loss + beta * kl_loss  (scalar).
            recon_loss: MSE reconstruction loss       (scalar, for logging).
            kl_loss:    KL divergence                 (scalar, for logging).
        """
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)

        # Sum over (C, H, W), mean over batch
        recon_loss = F.mse_loss(x_recon, x, reduction="sum") / x.size(0)

        # Sum over latent_dim, mean over batch
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
        ).mean()

        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def extra_repr(self) -> str:
        return (
            f"image_channels={self.image_channels}, "
            f"image_size={self.image_size}, "
            f"latent_dim={self.latent_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"enc_flat_dim={self._enc_flat_dim}"
        )