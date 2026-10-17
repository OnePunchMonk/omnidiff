"""The interface shared by continuous and discrete diffusion.

Time runs from t=1 (pure noise / fully masked) to t=0 (data). Each reverse step
``x_t -> x_next`` is a stochastic policy, and ``transition`` returns its exact
log-probability. Sampling and RL use the same call: pass ``x_next`` to score a
stored action instead of drawing a new one.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class Transition:
    x_next: torch.Tensor
    log_prob: torch.Tensor | None  # (B,), None when the step is deterministic
    stats: dict[str, torch.Tensor] = field(default_factory=dict)  # distribution parameters, used by kl()


class DiffusionProcess(ABC):
    @abstractmethod
    def timesteps(self, num_steps: int, device) -> torch.Tensor:
        """Descending grid of ``num_steps + 1`` times from 1 to 0."""

    @abstractmethod
    def prior(self, shape, cond: dict, device, generator=None) -> torch.Tensor:
        """The state at t=1."""

    @abstractmethod
    def training_loss(self, model, x0: torch.Tensor, cond: dict, drop_cond=None) -> torch.Tensor:
        """Denoising objective used for pretraining / SFT."""

    @abstractmethod
    def transition(self, pred, x_t, t, t_next, x_next=None, generator=None) -> Transition:
        """One reverse step from the model output ``pred``. Samples ``x_next`` unless it is given."""

    @abstractmethod
    def kl(self, new: Transition, ref: Transition) -> torch.Tensor:
        """KL(new || ref) between two policies' step distributions at the same state, (B,)."""

    def guide(self, cond_out: torch.Tensor, uncond_out: torch.Tensor, scale: float) -> torch.Tensor:
        return uncond_out + scale * (cond_out - uncond_out)


def reduce(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Per-sample reduction over every non-batch dim."""
    dims = tuple(range(1, x.dim()))
    if mode == "mean":
        return x.mean(dims)
    if mode == "sum":
        return x.sum(dims)
    raise ValueError(f"unknown reduction {mode!r}")


def gaussian_log_prob(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return -((x - mean) ** 2) / (2 * std**2) - torch.log(std) - 0.5 * math.log(2 * math.pi)


def expand_like(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return v.view(-1, *([1] * (x.dim() - 1)))


def randn(x: torch.Tensor, generator=None) -> torch.Tensor:
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
