from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..registry import PROCESSES
from .base import DiffusionProcess, Transition, expand_like, gaussian_log_prob, randn, reduce


@PROCESSES.register("gaussian")
class GaussianDiffusion(DiffusionProcess):
    """Continuous-time VP diffusion with epsilon prediction, sampled with DDIM(eta).

    ``eta=1`` is DDPM-like ancestral sampling, the policy DDPO (Black et al., 2023)
    optimises. ``eta=0`` is the deterministic DDIM ODE: fine for inference, but it
    has no log-probs, so RL refuses it.
    """

    def __init__(self, schedule: str = "vp_linear", eta: float = 1.0, clip_x0: float | None = None,
                 reduction: str = "mean", beta_min: float = 0.1, beta_max: float = 20.0, t_min: float = 1e-3):
        if schedule not in ("vp_linear", "cosine"):
            raise ValueError(f"unknown schedule {schedule!r}")
        self.schedule, self.eta, self.clip_x0, self.reduction = schedule, eta, clip_x0, reduction
        self.beta_min, self.beta_max, self.t_min = beta_min, beta_max, t_min

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        if self.schedule == "vp_linear":
            return torch.exp(-(self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2))
        s = 0.008
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        return (f / math.cos(s / (1 + s) * math.pi / 2) ** 2).clamp(min=1e-5)

    def timesteps(self, num_steps, device):
        return torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    def prior(self, shape, cond, device, generator=None):
        return torch.randn(shape, device=device, generator=generator)

    def training_loss(self, model, x0, cond, drop_cond=None):
        t = torch.rand(x0.shape[0], device=x0.device) * (1 - self.t_min) + self.t_min
        ab = expand_like(self.alpha_bar(t), x0)
        noise = torch.randn_like(x0)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        return F.mse_loss(model(x_t, t, cond, drop_cond), noise)

    def transition(self, pred, x_t, t, t_next, x_next=None, generator=None):
        ab_t, ab_s = self.alpha_bar(t), self.alpha_bar(t_next)
        x0 = (x_t - (1 - ab_t).sqrt() * pred) / ab_t.sqrt()
        if self.clip_x0 is not None:
            x0 = x0.clamp(-self.clip_x0, self.clip_x0)
        var = (self.eta**2 * (1 - ab_s) / (1 - ab_t) * (1 - ab_t / ab_s)).clamp(min=0)
        mean = ab_s.sqrt() * x0 + (1 - ab_s - var).clamp(min=0).sqrt() * pred
        std = var.sqrt()
        stats = {"mean": mean, "std": std}
        if float(std) == 0.0:
            return Transition(mean if x_next is None else x_next, None, stats)
        if x_next is None:
            x_next = mean + std * randn(mean, generator)
        return Transition(x_next, reduce(gaussian_log_prob(x_next, mean, std), self.reduction), stats)

    def kl(self, new, ref):
        diff = (new.stats["mean"] - ref.stats["mean"]) ** 2
        return reduce(diff / (2 * new.stats["std"] ** 2), self.reduction)
