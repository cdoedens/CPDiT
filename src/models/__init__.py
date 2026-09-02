"""
The CPDiT model: a score-based latent diffusion transformer for nowcasting.

Four modules, each with one job. Read them in this order:

    vae.py              Pixels <-> latents. A convolutional VAE that compresses
                        each (C, 256, 256) frame to a (4, 32, 32) latent map, so
                        the diffusion process runs in a space 64x smaller.

    networks.py         The two learned networks:
                          ContextEncoder — temporal transformer summarising the
                                           context frames
                          DiTDenoiser    — the score network, a Diffusion
                                           Transformer with 2-D neighbourhood
                                           attention and adaLN-Zero

    diffusion.py        The score-based process, with no learned parameters:
                        the forward SDEs (VP / cosine-VP / VE) and the
                        reverse-time samplers (predictor-corrector, PF-ODE).

    latent_diffusion.py The assembly. Owns the three pieces above, defines
                        score(), the denoising-score-matching objective, and
                        sample(). This is the only class training and inference
                        need to touch.
"""

from .diffusion import (
    SDE,
    CosineVPSDE,
    VESDE,
    VPSDE,
    build_sde,
    ode_sampler,
    pc_sampler,
)
from .latent_diffusion import LatentDiffusionTransformer
from .networks import (
    ContextEncoder,
    DiTBlock,
    DiTDenoiser,
    NeighbourhoodAttention2D,
    neighbourhood_mask,
)
from .vae import VariationalAutoencoder

__all__ = [
    # Assembly — the entry point.
    "LatentDiffusionTransformer",
    # Components.
    "VariationalAutoencoder",
    "ContextEncoder",
    "DiTDenoiser",
    "DiTBlock",
    "NeighbourhoodAttention2D",
    "neighbourhood_mask",
    # Diffusion process.
    "SDE",
    "VPSDE",
    "CosineVPSDE",
    "VESDE",
    "build_sde",
    "pc_sampler",
    "ode_sampler",
]
