"""Utility functions and loss definitions for diffusion models.

This module contains sampling utilities and loss implementations used in the
paper *"Elucidating the Design Space of Diffusion-Based Generative Models"*.
All public functions are annotated with PEP-484 type hints and documented using
Google style docstrings for clarity.
"""

from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from scipy.stats import betaprime
#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

randn_like=torch.randn_like

SIGMA_MIN=0.002
SIGMA_MAX=80
rho=7
S_churn= 1
S_min=0
S_max=float('inf')
S_noise=1


def sample(
    net: Any,
    num_samples: int,
    dim: int,
    num_steps: int = 50,
    device: str = "cuda:0",
) -> torch.Tensor:
    """Generate samples using the stochastic sampling scheme.

    Args:
        net: Denoising network with ``sigma_min``/``sigma_max`` attributes and a
            ``round_sigma`` method.
        num_samples: Number of latent samples to generate.
        dim: Dimensionality of the latent space.
        num_steps: Number of denoising steps to perform.
        device: Device on which to allocate tensors.

    Returns:
        A tensor containing generated samples.
    """
    # Start from Gaussian noise in latent space.
    latents = torch.randn([num_samples, dim], device=device)

    # Indices of diffusion steps.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    # Compute noise schedule.
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # Scale latents by initial noise level.
    x_next = latents.to(torch.float32) * t_steps[0]

    # Iteratively denoise.
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next)

    return x_next


def sample_step(
    net: Any,
    num_steps: int,
    i: int,
    t_cur: torch.Tensor,
    t_next: torch.Tensor,
    x_next: torch.Tensor,
) -> torch.Tensor:
    """Perform a single denoising step.

    Args:
        net: Denoising network.
        num_steps: Total number of sampling steps.
        i: Current step index.
        t_cur: Current noise level.
        t_next: Next noise level.
        x_next: Current latent sample.

    Returns:
        Updated latent sample after one step.
    """

    x_cur = x_next
    # Increase noise temporarily for better exploration.
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

    # Euler step to denoise.
    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction for improved accuracy.
    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

class VPLoss:
    """Variance Preserving loss used for diffusion training."""

    def __init__(self, beta_d: float = 19.9, beta_min: float = 0.1, epsilon_t: float = 1e-5) -> None:
        """Initialize VPLoss parameters.

        Args:
            beta_d: Beta distribution parameter controlling noise growth.
            beta_min: Minimum beta value.
            epsilon_t: Minimum time for noise scheduling.
        """
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(
        self,
        denosie_fn: Callable[..., torch.Tensor],
        data: torch.Tensor,
        labels: torch.Tensor,
        augment_pipe: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
    ) -> torch.Tensor:
        """Compute the VP loss.

        Args:
            denosie_fn: Denoising function.
            data: Input tensor.
            labels: Conditioning labels.
            augment_pipe: Optional augmentation pipeline.

        Returns:
            Loss tensor for each sample.
        """
        rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
        n = torch.randn_like(y) * sigma
        D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)
        return weight * ((D_yn - y) ** 2)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the noise level from a time value.

        Args:
            t: Time parameter controlling the noise schedule.

        Returns:
            Noise level ``sigma`` derived from ``t``.
        """
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VELoss:
    """Variance Exploding loss for diffusion training."""

    def __init__(
        self,
        sigma_min: float = 0.02,
        sigma_max: float = 100,
        D: int = 128,
        N: int = 3072,
        opts: Optional[Any] = None,
    ) -> None:
        """Initialize VELoss parameters.

        Args:
            sigma_min: Minimum noise level.
            sigma_max: Maximum noise level.
            D: Dimensionality of the hidden representation.
            N: Dimensionality of the augmentation space.
            opts: Optional configuration object.
        """

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.D = D
        self.N = N
        self.opts = opts
        print(f"In VE loss: D:{self.D}, N:{self.N}")

    def __call__(
        self,
        denosie_fn: Callable[..., torch.Tensor],
        data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        augment_pipe: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
        stf: bool = False,
        pfgmpp: bool = False,
        ref_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the VE loss.

        This function supports the PFGM++ sampler when ``pfgmpp`` is ``True``.

        Args:
            denosie_fn: Denoising function.
            data: Input tensor.
            labels: Optional conditioning labels.
            augment_pipe: Optional augmentation pipeline.
            stf: Unused flag kept for backward compatibility.
            pfgmpp: Whether to use the PFGM++ perturbation scheme.
            ref_data: Optional reference data for certain samplers.

        Returns:
            Loss tensor for each sample.
        """
        if pfgmpp:

            # Sample sigma from log-uniform distribution.
            rnd_uniform = torch.rand(data.shape[0], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)

            r = sigma.double() * np.sqrt(self.D).astype(np.float64)
            # Sample from inverse-beta distribution.
            samples_norm = np.random.beta(a=self.N / 2.0, b=self.D / 2.0, size=data.shape[0]).astype(np.double)

            samples_norm = np.clip(samples_norm, 1e-3, 1 - 1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
            inverse_beta = torch.from_numpy(inverse_beta).to(data.device).double()
            # Sampling from p_r(R) by change-of-variable.
            samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
            samples_norm = samples_norm.view(len(samples_norm), -1)
            # Uniformly sample the angular direction.
            gaussian = torch.randn(data.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            # Construct the perturbation for x.
            perturbation_x = unit_gaussian * samples_norm
            perturbation_x = perturbation_x.float()

            sigma = sigma.reshape((len(sigma), 1, 1, 1))
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = perturbation_x.view_as(y)
            D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)
        else:
            rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = torch.randn_like(y) * sigma
            D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)

        return weight * ((D_yn - y) ** 2)

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

class EDMLoss:
    """Improved loss function from the EDM paper."""

    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        hid_dim: int = 100,
        gamma: float = 5,
        opts: Optional[Any] = None,
    ) -> None:
        """Initialize the EDM loss.

        Args:
            P_mean: Mean of the log-normal noise distribution.
            P_std: Standard deviation of the log-normal noise distribution.
            sigma_data: Expected data standard deviation.
            hid_dim: Hidden dimensionality used by the model.
            gamma: Loss scaling parameter.
            opts: Optional configuration object.
        """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts

    def __call__(self, denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], data: torch.Tensor) -> torch.Tensor:
        """Compute the EDM loss.

        Args:
            denoise_fn: Network used for denoising.
            data: Input tensor.

        Returns:
            Loss tensor for each sample.
        """

        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        D_yn = denoise_fn(y + n, sigma)

        target = y
        return weight.unsqueeze(1) * ((D_yn - target) ** 2)

