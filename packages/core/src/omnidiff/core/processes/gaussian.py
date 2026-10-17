from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..registry import PROCESSES
from .base import ContinuousProcess, Transition, bcast, expand_like, gaussian_log_prob, randn, reduce


@PROCESSES.register("gaussian")
class GaussianDiffusion(ContinuousProcess):
    """Continuous-time VP diffusion with epsilon prediction.

    Samplers:
      ``ddim``: DDIM(eta). ``eta=1`` is DDPM-like ancestral sampling, the policy DDPO
      (Black et al., 2023) optimises; ``eta=0`` is the deterministic ODE.
      ``renoise``: predict x0, re-noise to the next time (few-step students).
      ``ode``: the deterministic DDIM step.
    """

    def __init__(self, schedule: str = "vp_linear", eta: float = 1.0, clip_x0: float | None = None,
                 reduction: str = "mean", beta_min: float = 0.1, beta_max: float = 20.0, t_min: float = 1e-3,
                 sampler: str = "ddim"):
        if schedule not in ("vp_linear", "cosine"):
            raise ValueError(f"unknown schedule {schedule!r}")
        if sampler not in ("ddim", "renoise", "ode"):
            raise ValueError(f"unknown sampler {sampler!r}")
        self.schedule, self.eta, self.clip_x0, self.reduction = schedule, eta, clip_x0, reduction
        self.beta_min, self.beta_max, self.t_min, self.sampler = beta_min, beta_max, t_min, sampler

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        if self.schedule == "vp_linear":
            return torch.exp(-(self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2))
        s = 0.008
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        return (f / math.cos(s / (1 + s) * math.pi / 2) ** 2).clamp(min=1e-5)

    def alpha_sigma(self, t):
        ab = self.alpha_bar(t)
        return ab.sqrt(), (1 - ab).sqrt()

    def to_x0(self, pred, x_t, t):
        a, s = self.alpha_sigma(bcast(t, x_t))
        x0 = (x_t - s * pred) / a
        return x0 if self.clip_x0 is None else x0.clamp(-self.clip_x0, self.clip_x0)

    def to_eps(self, pred, x_t, t):
        return pred

    def timesteps(self, num_steps, device):
        return torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    def prior(self, shape, cond, device, generator=None):
        return torch.randn(shape, device=device, generator=generator)

    def training_loss(self, model, x0, cond, drop_cond=None):
        t = torch.rand(x0.shape[0], device=x0.device) * (1 - self.t_min) + self.t_min
        noise = torch.randn_like(x0)
        return F.mse_loss(model(self.q_sample(x0, t, noise), t, cond, drop_cond), noise)

    def transition(self, pred, x_t, t, t_next, x_next=None, generator=None):
        if self.sampler == "renoise":
            return self.renoise_transition(pred, x_t, t, t_next, x_next, generator)
        if self.sampler == "ode":
            return self.ode_transition(pred, x_t, t, t_next, x_next)
        ab_t, ab_s = self.alpha_bar(t), self.alpha_bar(t_next)
        x0 = self.to_x0(pred, x_t, t)
        var = (self.eta**2 * (1 - ab_s) / (1 - ab_t) * (1 - ab_t / ab_s)).clamp(min=0)
        mean = ab_s.sqrt() * x0 + (1 - ab_s - var).clamp(min=0).sqrt() * pred
        std = var.sqrt()
        stats = {"mean": mean, "std": std}
        if float(std) == 0.0:
            return Transition(mean if x_next is None else x_next, None, stats)
        if x_next is None:
            x_next = mean + std * randn(mean, generator)
        return Transition(x_next, reduce(gaussian_log_prob(x_next, mean, std), self.reduction), stats)
