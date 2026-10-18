from __future__ import annotations

import torch
import torch.nn.functional as F

from ..registry import PROCESSES
from .base import ContinuousProcess, Transition, bcast, gaussian_log_prob, randn, reduce


@PROCESSES.register("flow")
class FlowMatching(ContinuousProcess):
    """Rectified flow ``x_t = (1 - t) x0 + t * noise`` with velocity target ``noise - x0``.

    Samplers:
      ``sde`` (with ``noise_level > 0``): the marginal-preserving reverse SDE from
      Flow-GRPO (Liu et al., 2025),

        dx = [v + sigma_t^2 / (2t) * (x + (1 - t) v)] dt + sigma_t dW,  sigma_t = a * sqrt(t / (1 - t)),

      which has the ODE's marginals but makes every step a Gaussian policy.
      ``ode``: Euler, ``x + (t_next - t) v``. With a two-time model the network output is
      the average velocity over [t_next, t], and this same step is the flow map.
      ``renoise``: predict x0, re-noise to the next time (few-step students).
    ``noise_level = 0`` with ``sampler: sde`` falls back to ``ode``.
    """

    def __init__(self, noise_level: float = 0.7, shift: float = 1.0, time_sampling: str = "logit_normal",
                 reduction: str = "mean", sampler: str = "sde"):
        if time_sampling not in ("logit_normal", "uniform"):
            raise ValueError(f"unknown time_sampling {time_sampling!r}")
        if sampler not in ("sde", "ode", "renoise"):
            raise ValueError(f"unknown sampler {sampler!r}")
        self.noise_level, self.shift, self.time_sampling, self.reduction = noise_level, shift, time_sampling, reduction
        self.sampler = "ode" if sampler == "sde" and noise_level <= 0 else sampler

    def alpha_sigma(self, t):
        return 1 - t, t

    def to_x0(self, pred, x_t, t):
        return x_t - bcast(t, x_t) * pred

    def to_eps(self, pred, x_t, t):
        return x_t + (1 - bcast(t, x_t)) * pred

    def timesteps(self, num_steps, device):
        t = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        return self.shift * t / (1 + (self.shift - 1) * t)

    def sample_time(self, n: int, device) -> torch.Tensor:
        if self.time_sampling == "logit_normal":
            return torch.sigmoid(torch.randn(n, device=device))
        return torch.rand(n, device=device)

    def prior(self, shape, cond, device, generator=None):
        return torch.randn(shape, device=device, generator=generator)

    def training_loss(self, model, x0, cond, drop_cond=None):
        t = self.sample_time(x0.shape[0], x0.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(model(self.q_sample(x0, t, noise), t, cond, drop_cond), noise - x0)

    def transition(self, pred, x_t, t, t_next, x_next=None, generator=None):
        if self.sampler == "renoise":
            return self.renoise_transition(pred, x_t, t, t_next, x_next, generator)
        if self.sampler == "ode":
            return self.ode_transition(pred, x_t, t, t_next, x_next)
        dt = t_next - t
        # 1 / (1 - t) blows up at t = 1; like Flow-GRPO, evaluate it at the next time there.
        sigma = self.noise_level * (t / (1 - torch.where(t >= 1, t_next, t))).sqrt()
        drift = pred + sigma**2 / (2 * t) * (x_t + (1 - t) * pred)
        mean = x_t + drift * dt
        std = sigma * (-dt).sqrt()
        if x_next is None:
            x_next = mean + std * randn(mean, generator)
        stats = {"mean": mean, "std": std}
        return Transition(x_next, reduce(gaussian_log_prob(x_next, mean, std), self.reduction), stats)
