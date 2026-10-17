from __future__ import annotations

import torch
import torch.nn.functional as F

from ..registry import PROCESSES
from .base import DiffusionProcess, Transition, expand_like, gaussian_log_prob, randn, reduce


@PROCESSES.register("flow")
class FlowMatching(DiffusionProcess):
    """Rectified flow ``x_t = (1 - t) x0 + t * noise`` with velocity target ``noise - x0``.

    With ``noise_level > 0`` sampling uses the marginal-preserving reverse SDE from
    Flow-GRPO (Liu et al., 2025),

        dx = [v + sigma_t^2 / (2t) * (x + (1 - t) v)] dt + sigma_t dW,  sigma_t = a * sqrt(t / (1 - t)),

    which has the same marginals as the ODE but makes every step a Gaussian policy.
    ``noise_level = 0`` is the plain Euler ODE (inference only).
    """

    def __init__(self, noise_level: float = 0.7, shift: float = 1.0, time_sampling: str = "logit_normal",
                 reduction: str = "mean"):
        if time_sampling not in ("logit_normal", "uniform"):
            raise ValueError(f"unknown time_sampling {time_sampling!r}")
        self.noise_level, self.shift, self.time_sampling, self.reduction = noise_level, shift, time_sampling, reduction

    def timesteps(self, num_steps, device):
        t = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        return self.shift * t / (1 + (self.shift - 1) * t)

    def prior(self, shape, cond, device, generator=None):
        return torch.randn(shape, device=device, generator=generator)

    def training_loss(self, model, x0, cond, drop_cond=None):
        B = x0.shape[0]
        if self.time_sampling == "logit_normal":
            t = torch.sigmoid(torch.randn(B, device=x0.device))
        else:
            t = torch.rand(B, device=x0.device)
        noise = torch.randn_like(x0)
        x_t = (1 - expand_like(t, x0)) * x0 + expand_like(t, x0) * noise
        return F.mse_loss(model(x_t, t, cond, drop_cond), noise - x0)

    def transition(self, pred, x_t, t, t_next, x_next=None, generator=None):
        dt = t_next - t
        if self.noise_level <= 0:
            mean = x_t + pred * dt
            return Transition(mean if x_next is None else x_next, None, {"mean": mean})
        # 1 / (1 - t) blows up at t = 1; like Flow-GRPO, evaluate it at the next time there.
        sigma = self.noise_level * (t / (1 - torch.where(t >= 1, t_next, t))).sqrt()
        drift = pred + sigma**2 / (2 * t) * (x_t + (1 - t) * pred)
        mean = x_t + drift * dt
        std = sigma * (-dt).sqrt()
        if x_next is None:
            x_next = mean + std * randn(mean, generator)
        stats = {"mean": mean, "std": std}
        return Transition(x_next, reduce(gaussian_log_prob(x_next, mean, std), self.reduction), stats)

    def kl(self, new, ref):
        diff = (new.stats["mean"] - ref.stats["mean"]) ** 2
        return reduce(diff / (2 * new.stats["std"] ** 2), self.reduction)
