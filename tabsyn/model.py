from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor

from tabsyn.diffusion_utils import EDMLoss

ModuleType = Union[str, Callable[..., nn.Module]]


class SiLU(nn.Module):
    """Sigmoid Linear Unit activation."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply the SiLU activation."""
        return x * torch.sigmoid(x)


class PositionalEmbedding(torch.nn.Module):
    """Generate sine/cosine positional embeddings."""

    def __init__(self, num_channels: int, max_positions: int = 10000, endpoint: bool = False) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x: Tensor) -> Tensor:
        """Compute positional embeddings for input ``x``."""
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


def reglu(x: Tensor) -> Tensor:
    """Apply the ReGLU activation [Shazeer, 2020]."""

    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """Apply the GEGLU activation [Shazeer, 2020]."""

    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """Module wrapper around :func:`reglu`."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply ReGLU to ``x``."""
        return reglu(x)


class GEGLU(nn.Module):
    """Module wrapper around :func:`geglu`."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply GEGLU to ``x``."""
        return geglu(x)


class FourierEmbedding(torch.nn.Module):
    """Random Fourier feature positional embedding."""

    def __init__(self, num_channels: int, scale: int = 16) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x: Tensor) -> Tensor:
        """Compute Fourier embeddings for ``x``."""
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLPDiffusion(nn.Module):
    """Simple MLP-based denoising network."""

    def __init__(self, d_in: int, dim_t: int = 512) -> None:
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )

    def forward(self, x: Tensor, noise_labels: Tensor, class_labels: Optional[Tensor] = None) -> Tensor:
        """Denoise ``x`` conditioned on noise labels."""
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        emb = self.time_embed(emb)

        x = self.proj(x) + emb
        return self.mlp(x)


class Precond(nn.Module):
    """Wrapper that applies EDM preconditioning."""

    def __init__(
        self,
        denoise_fn: Callable[[Tensor, Tensor], Tensor],
        hid_dim: int,
        sigma_min: float = 0,
        sigma_max: float = float("inf"),
        sigma_data: float = 0.5,
    ) -> None:
        super().__init__()

        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        # Wrapped denoising network.
        self.denoise_fn_F = denoise_fn

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Apply preconditioning to ``x`` at noise level ``sigma``."""

        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten())

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma: Tensor) -> Tensor:
        """Round ``sigma`` to the nearest supported value."""
        return torch.as_tensor(sigma)


class Model(nn.Module):
    """EDM-based diffusion model wrapper."""

    def __init__(
        self,
        denoise_fn: Callable[[Tensor, Tensor], Tensor],
        hid_dim: int,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        gamma: float = 5,
        opts: Optional[Any] = None,
        pfgmpp: bool = False,
    ) -> None:
        super().__init__()

        self.denoise_fn_D = Precond(denoise_fn, hid_dim)
        self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None)

    def forward(self, x: Tensor) -> Tensor:
        """Compute the EDM loss for input ``x``."""

        loss = self.loss_fn(self.denoise_fn_D, x)
        return loss.mean(-1).mean()
