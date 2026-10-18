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


def bcast(t, x: torch.Tensor) -> torch.Tensor:
    """A scalar or per-sample (B,) time, shaped to broadcast against ``x``."""
    t = torch.as_tensor(t, device=x.device, dtype=torch.float32)
    return t if t.dim() == 0 else expand_like(t, x)


class ContinuousProcess(DiffusionProcess):
    """Processes with ``x_t = alpha(t) x0 + sigma(t) noise``.

    Subclasses define the schedule (``alpha_sigma``) and how the model output maps
    to x0 / noise. Everything distillation needs follows from those: forward
    noising, the deterministic probability-flow step, and the x0-renoise step used
    by consistency and DMD-style few-step students. Times may be scalars or (B,).
    """

    sampler: str

    @abstractmethod
    def alpha_sigma(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @abstractmethod
    def to_x0(self, pred, x_t, t) -> torch.Tensor:
        ...

    @abstractmethod
    def to_eps(self, pred, x_t, t) -> torch.Tensor:
        ...

    def q_sample(self, x0, t, noise) -> torch.Tensor:
        a, s = self.alpha_sigma(bcast(t, x0))
        return a * x0 + s * noise

    def ode_step(self, pred, x_t, t, t_next) -> torch.Tensor:
        """Deterministic probability-flow step (DDIM with eta=0 / Euler for rectified flow)."""
        return self.q_sample(self.to_x0(pred, x_t, t), t_next, self.to_eps(pred, x_t, t))

    def renoise_transition(self, pred, x_t, t, t_next, x_next=None, generator=None) -> Transition:
        """Jump to the predicted x0, then re-noise it to ``t_next`` with fresh noise.

        This is the multistep sampler of consistency models and DMD2. It is a Gaussian
        policy, ``N(alpha(s) x0_hat, sigma(s)^2 I)``, so few-step students trained with
        it can be RL-tuned with exact log-probs. The final step (sigma = 0) is deterministic.
        """
        x0 = self.to_x0(pred, x_t, t)
        a, std = self.alpha_sigma(torch.as_tensor(t_next, device=x_t.device, dtype=torch.float32))
        mean = a * x0
        stats = {"mean": mean, "std": std}
        if float(std) == 0.0:
            return Transition(mean if x_next is None else x_next, None, stats)
        if x_next is None:
            x_next = mean + std * randn(mean, generator)
        return Transition(x_next, reduce(gaussian_log_prob(x_next, mean, std), self.reduction), stats)

    def ode_transition(self, pred, x_t, t, t_next, x_next=None) -> Transition:
        mean = self.ode_step(pred, x_t, t, t_next)
        return Transition(mean if x_next is None else x_next, None, {"mean": mean})

    def kl(self, new, ref):
        diff = (new.stats["mean"] - ref.stats["mean"]) ** 2
        return reduce(diff / (2 * new.stats["std"] ** 2), self.reduction)


def randn(x: torch.Tensor, generator=None) -> torch.Tensor:
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
