"""
The score-based diffusion process: forward SDEs and reverse-time samplers.

This module holds the *mathematics* of the diffusion process and contains no
learned parameters. It knows nothing about satellites, VAEs or transformers —
every function here takes a score function ``s(x, t)`` and works with it
abstractly, which is what keeps the model assembly in `latent_diffusion.py`
readable and this file independently testable.

Where this sits in the workflow
-------------------------------
    latent_diffusion.py  owns the model and supplies score_fn
              |
              v
    +--------------------------------------------------+
    |  THIS FILE                                        |
    |                                                   |
    |  TRAINING     sde.perturb(x0, t) -> x_t, z, sigma |
    |               (the DSM target -z/sigma is exact)  |
    |                                                   |
    |  SAMPLING     pc_sampler   reverse SDE + Langevin |
    |               ode_sampler  probability-flow ODE   |
    +--------------------------------------------------+

Forward / reverse processes
---------------------------
    forward :  dx = f(x, t) dt + g(t) dw
    reverse :  dx = [f(x, t) - g(t)^2 * score(x, t)] dt + g(t) dw_bar
    PF ODE  :  dx = [f(x, t) - 0.5 * g(t)^2 * score(x, t)] dt

Every SDE here has a Gaussian perturbation kernel

    p(x_t | x_0) = N(alpha(t) * x_0, sigma(t)^2 I)

which is the whole point: the score of that kernel is known exactly,
``-(x_t - alpha*x_0) / sigma^2 = -z/sigma``, so the intractable score-matching
objective collapses into a regression onto it (denoising score matching).

Three SDEs are provided:

  VPSDE       Variance Preserving, linear beta(t). The continuous-time limit of
              DDPM — the closest analogue of the previous discrete model.
  CosineVPSDE Variance Preserving with the cosine alpha_bar of Nichol & Dhariwal,
              expressed continuously. The project default.
  VESDE       Variance Exploding (the original NCSN / score-matching schedule).
              Never attenuates the signal; noise grows geometrically instead.

References:
  - Song et al. 2021   — https://arxiv.org/abs/2011.13456 (score-based SDEs)
  - Song & Ermon 2019  — https://arxiv.org/abs/1907.05600 (NCSN, VE)
  - Vincent 2011       — denoising score matching
  - Anderson 1982      — the reverse-time SDE
"""

from __future__ import annotations

import abc
import math
from typing import Callable, Optional

import torch


# =============================================================================
# 1. Forward SDEs
#
#    Each defines a noising process with a closed-form Gaussian kernel, so
#    training never has to simulate the forward process step by step.
# =============================================================================

def broadcast_to(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a per-sample vector (B,) so it broadcasts against x (B, ...)."""
    return v.reshape(-1, *([1] * (x.ndim - 1)))


class SDE(abc.ABC):
    """
    Base class for a forward SDE with a Gaussian perturbation kernel.

    All subclasses must implement:
        - sde(x, t) -> f, g
        - alpha_sigma(t) -> alpha, sigma
    ...and the will contain derived methods:
        - marginal_prob(x, t) -> mean, std
        - perturb(x0, t) -> x_t, noise, sigma
    
    """

    def __init__(self, T: float = 1.0, t_eps: float = 1e-3):
        # Sampling and training both stop short of t=0: sigma(0) = 0 makes the
        # score singular, so the objective and the reverse chain are defined on
        # [t_eps, T].
        self.T     = float(T)
        self.t_eps = float(t_eps)

    # -- forward process ------------------------------------------------- #

    @abc.abstractmethod
    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
        - Drift f(x, t) shaped like x
        - diffusion g(t) shaped (B,)
        where dx = f(x,t)dt + g(t)dw, i.e. the forward SDE
        """

    @abc.abstractmethod
    def alpha_sigma(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        returns the Perturbation-kernel coefficients:
        p(x_t|x_0) = N(alpha*x_0, sigma^2 I)
        
        i.e. alpha(t) and sigma(t)
        """

    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mean and (broadcast) std of p(x_t | x_0 = x).

        Turns alpha(t) and sigma(t) into the actual mean and std of the pertubation distribution
        """
        alpha, sigma = self.alpha_sigma(t)
        return broadcast_to(alpha, x) * x, broadcast_to(sigma, x)

    def perturb(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perturb a clean latent x_0 at time t, i.e. the "perturbation-kernel", using marginal_prob.
        Instead of simulating the SDE (x0 -> x1 -> ... -> x_t), we can sample directly from the known Gaussian kernel
        x_t = alpha(t) * x_0, sigma(t) * z

        Inputs:
            x0:    clean latent (B, C, H, W)
            t:     time (B,)
            noise: optional noise to use instead of sampling it
        Returns:
            x_t: perturbed latent (B, C, H, W)
            noise: sampled noise (B, C, H, W)
            sigma_broadcast: broadcast sigma (B, 1, 1, 1)
        The score of the perturbation kernel is -noise / sigma, which is the Denoising Score Matching regression target.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        mean, sigma = self.marginal_prob(x0, t)
        return mean + sigma * noise, noise, sigma

    # -- reverse process -------------------------------------------------- #

    def reverse_sde(
        self, x: torch.Tensor, t: torch.Tensor, score: torch.Tensor,
        probability_flow: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Drift and diffusion of the reverse-time SDE (Anderson 1982).

        Forward SDE is:
            dx = f*dt + g*dw, where f is drift
        Anderson (1982) reverse SDE is:
            dx = [f-(g**2)*s(x,t)]*dt + gdw, where s(x,t) the score function approximated by -noise / sigma

        With ``probability_flow=True`` the diffusion term is dropped and the
        drift halved, giving the deterministic probability-flow ODE that shares
        the same marginals.
        """
        drift, g = self.sde(x, t)
        g2 = broadcast_to(g, x) ** 2
        coeff = 0.5 if probability_flow else 1.0
        drift = drift - coeff * g2 * score
        if probability_flow:
            return drift, torch.zeros_like(g)
        return drift, g

    @abc.abstractmethod
    def prior_sampling(self, shape, device, generator=None) -> torch.Tensor:
        """Draw from the tractable prior p_T the reverse chain starts at."""


class VPSDE(SDE):
    """
    Variance Preserving SDE with linear beta(t) — the continuous limit of DDPM.

        f(x, t) = -0.5 * beta(t) * x
        g(t)    = sqrt(beta(t))
        beta(t) = beta_min + t * (beta_max - beta_min)

    Integrating the drift gives a closed-form kernel:
        alpha(t) = exp(-0.25 t^2 (beta_max - beta_min) - 0.5 t beta_min)
        sigma(t) = sqrt(1 - alpha(t)^2)
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0,
                 T: float = 1.0, t_eps: float = 1e-3):
        super().__init__(T=T, t_eps=t_eps)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, x, t):
        beta = self.beta(t)
        return -0.5 * broadcast_to(beta, x) * x, torch.sqrt(beta)

    def alpha_sigma(self, t):
        log_alpha = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        alpha = torch.exp(log_alpha)
        return alpha, torch.sqrt((1.0 - alpha ** 2).clamp(min=1e-10))

    def prior_sampling(self, shape, device, generator=None):
        return torch.randn(shape, device=device, generator=generator)


class CosineVPSDE(SDE):
    """
    Variance Preserving SDE whose alpha_bar follows the cosine schedule.

        alpha_bar(t) = cos^2(u(t)) / cos^2(u(0)),  u(t) = (t + s)/(1 + s) * pi/2
        alpha(t)     = sqrt(alpha_bar(t)),  sigma(t) = sqrt(1 - alpha_bar(t))

    The drift follows from beta(t) = -d/dt log alpha_bar(t) = pi/(1+s) * tan(u)
    """

    def __init__(self, s: float = 0.008, beta_max: float = 999.0,
                 T: float = 1.0, t_eps: float = 1e-3):
        super().__init__(T=T, t_eps=t_eps)
        self.s = float(s)
        self.beta_max = float(beta_max)
        self._u0 = (self.s / (1.0 + self.s)) * math.pi / 2.0
        self._log_cos_u0 = math.log(math.cos(self._u0))

    def _u(self, t: torch.Tensor) -> torch.Tensor:
        return (t + self.s) / (1.0 + self.s) * (math.pi / 2.0)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        u = self._u(t).clamp(max=math.pi / 2.0 - 1e-6)
        return (math.pi / (1.0 + self.s) * torch.tan(u)).clamp(max=self.beta_max)

    def sde(self, x, t):
        beta = self.beta(t)
        return -0.5 * broadcast_to(beta, x) * x, torch.sqrt(beta)

    def alpha_sigma(self, t):
        u = self._u(t).clamp(max=math.pi / 2.0 - 1e-6)
        log_alpha_bar = 2.0 * (torch.log(torch.cos(u)) - self._log_cos_u0)
        alpha_bar = torch.exp(log_alpha_bar).clamp(min=1e-10, max=1.0)
        return torch.sqrt(alpha_bar), torch.sqrt((1.0 - alpha_bar).clamp(min=1e-10))

    def prior_sampling(self, shape, device, generator=None):
        return torch.randn(shape, device=device, generator=generator)


class VESDE(SDE):
    """
    Variance Exploding SDE (NCSN). The signal is never attenuated; noise grows
    geometrically instead.

        alpha(t) = 1
        sigma(t) = sigma_min * (sigma_max / sigma_min)^t
        f(x, t)  = 0
        g(t)     = sigma(t) * sqrt(2 log(sigma_max / sigma_min))
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0,
                 T: float = 1.0, t_eps: float = 1e-5):
        super().__init__(T=T, t_eps=t_eps)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self._log_ratio = math.log(self.sigma_max / self.sigma_min)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sde(self, x, t):
        sigma = self.sigma(t)
        return torch.zeros_like(x), sigma * math.sqrt(2.0 * self._log_ratio)

    def alpha_sigma(self, t):
        return torch.ones_like(t), self.sigma(t)

    def prior_sampling(self, shape, device, generator=None):
        return torch.randn(shape, device=device, generator=generator) * self.sigma_max


_SDES = {"vp": VPSDE, "vp_cosine": CosineVPSDE, "ve": VESDE}


def build_sde(name: str = "vp_cosine", **kwargs) -> SDE:
    """Construct an SDE by config name, ignoring kwargs it does not accept."""
    key = str(name).lower()
    if key not in _SDES:
        raise ValueError(f"Unknown sde '{name}'. Options: {sorted(_SDES)}")
    cls = _SDES[key]
    import inspect
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return cls(**{k: v for k, v in kwargs.items() if k in accepted and v is not None})


# =============================================================================
# 2. Reverse-time samplers
#
#    Both integrate from t = T down to t = t_eps using only score_fn(x, t).
#    Neither knows anything about the model that produced the score.
#    ONLY USED IN SAMPLING, NOT TRAINING.
# =============================================================================

ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _timesteps(sde: SDE, num_steps: int, device) -> torch.Tensor:
    """Descending time grid from T to t_eps."""
    return torch.linspace(sde.T, sde.t_eps, num_steps + 1, device=device)


@torch.no_grad()
def pc_sampler(
    sde:              SDE,
    score_fn:         ScoreFn, # supplied by learned DiT
    shape:            tuple,
    device:           torch.device,
    num_steps:        int   = 100,
    corrector_steps:  int   = 1,
    snr:              float = 0.16,
    probability_flow: bool  = False,
    denoise:          bool  = True,
    clamp_value:      Optional[float] = None,
    generator:        Optional[torch.Generator] = None,
    x_T:              Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Predictor-Corrector sampler. Combines a reverse-SDE step (predictor) to
    move toward clearner sample, with an annealed Langevin step (corrector)
    to improve sample quality by moving along the score function:

    xT -> corrector -> predictor -> corrector -> predictor -> ... -> x0

    Args:
        num_steps:       reverse-SDE (predictor) steps.
        corrector_steps: Langevin corrector steps per predictor step. 0 gives a
                         pure Euler-Maruyama reverse-SDE sampler.
        snr:             target signal-to-noise ratio for the Langevin step
                         size. 0.16 is the value tuned in Song et al.; the step
                         size adapts to the score magnitude, so this transfers
                         across noise levels.
        denoise:         apply one final Tweedie step at t_eps, which removes
                         the residual noise the discretisation leaves behind.
        clamp_value:     optional bound on the iterate, as a divergence guard.
    """
    ts = _timesteps(sde, num_steps, device)
    dt = -(sde.T - sde.t_eps) / num_steps          # negative: integrating backwards

    x = sde.prior_sampling(shape, device, generator) if x_T is None else x_T.clone()

    for i in range(num_steps):
        t = torch.full((shape[0],), float(ts[i]), device=device)

        # ---- Corrector: annealed Langevin at fixed t --------------------- #
        for _ in range(corrector_steps):
            score = score_fn(x, t)
            noise = torch.randn(x.shape, device=device, generator=generator)
            # Step size from the score/noise norm ratio (Song et al. Alg. 4/5).
            grad_norm  = score.reshape(shape[0], -1).norm(dim=-1).mean()
            noise_norm = noise.reshape(shape[0], -1).norm(dim=-1).mean()
            step = 2.0 * (snr * noise_norm / grad_norm.clamp(min=1e-12)) ** 2
            x = x + step * score + torch.sqrt(2.0 * step) * noise
            if clamp_value is not None:
                x = x.clamp(-clamp_value, clamp_value)

        # ---- Predictor: one reverse-SDE / PF-ODE step -------------------- #
        # Uses Euler-Maruyama when probability_flow=False, and Euler when probability_flow=True.
        score = score_fn(x, t)
        drift, g = sde.reverse_sde(x, t, score, probability_flow=probability_flow)
        x = x + drift * dt
        if not probability_flow:
            noise = torch.randn(x.shape, device=device, generator=generator)
            x = x + broadcast_to(g, x) * torch.sqrt(torch.tensor(-dt, device=device)) * noise
        if clamp_value is not None:
            x = x.clamp(-clamp_value, clamp_value)

    if denoise:
        # Tweedie: E[x_0 | x_t] = (x + sigma^2 * score) / alpha.
        t = torch.full((shape[0],), sde.t_eps, device=device)
        alpha, sigma = sde.alpha_sigma(t)
        score = score_fn(x, t)
        x = (x + broadcast_to(sigma, x) ** 2 * score) / broadcast_to(alpha, x).clamp(min=1e-8)
        if clamp_value is not None:
            x = x.clamp(-clamp_value, clamp_value)

    return x


@torch.no_grad()
def ode_sampler(
    sde:         SDE,
    score_fn:    ScoreFn,
    shape:       tuple,
    device:      torch.device,
    num_steps:   int = 100,
    method:      str = "heun",
    clamp_value: Optional[float] = None,
    generator:   Optional[torch.Generator] = None,
    x_T:         Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Deterministic probability-flow ODE sampler.

    ``method="heun"`` uses a second-order predictor-corrector step, which is
    markedly more accurate per function evaluation than Euler on the same grid
    (two score evaluations per step).
    """
    if method not in ("euler", "heun"):
        raise ValueError(f"method must be 'euler' or 'heun', got {method!r}")

    ts = _timesteps(sde, num_steps, device)
    x  = sde.prior_sampling(shape, device, generator) if x_T is None else x_T.clone()

    def drift_at(x_, t_scalar):
        t = torch.full((shape[0],), float(t_scalar), device=device)
        return sde.reverse_sde(x_, t, score_fn(x_, t), probability_flow=True)[0]

    for i in range(num_steps):
        t_cur, t_next = float(ts[i]), float(ts[i + 1])
        dt = t_next - t_cur                       # negative

        d = drift_at(x, t_cur)
        if method == "euler":
            x = x + d * dt
        else:
            x_euler = x + d * dt
            d2 = drift_at(x_euler, t_next)
            x = x + 0.5 * (d + d2) * dt
        if clamp_value is not None:
            x = x.clamp(-clamp_value, clamp_value)

    return x
