"""The sampling loop, shared by inference and RL rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .processes.base import DiffusionProcess, Transition


@dataclass
class Trajectory:
    xs: torch.Tensor          # (B, K + 1, *shape): prior at index 0, final sample at index K
    log_probs: torch.Tensor   # (B, K), zero on deterministic steps
    stochastic: torch.Tensor  # (K,) bool, steps with a defined log-prob
    timesteps: torch.Tensor   # (K + 1,)

    @property
    def samples(self) -> torch.Tensor:
        return self.xs[:, -1]

    @staticmethod
    def cat(trajs: list["Trajectory"]) -> "Trajectory":
        return Trajectory(
            torch.cat([t.xs for t in trajs]),
            torch.cat([t.log_probs for t in trajs]),
            trajs[0].stochastic,
            trajs[0].timesteps,
        )


def predict(model, x, t, cond, guidance_scale: float = 1.0, process: DiffusionProcess | None = None):
    """Model output at (x, t), with classifier-free guidance when the model supports it."""
    if guidance_scale == 1.0 or not getattr(model, "supports_guidance", False):
        return model(x, t, cond)
    B = x.shape[0]
    both = {k: torch.cat([v, v]) for k, v in cond.items()}
    drop = torch.cat([torch.zeros(B, dtype=torch.bool), torch.ones(B, dtype=torch.bool)]).to(x.device)
    c, u = model(torch.cat([x, x]), t, both, drop).chunk(2)
    return process.guide(c, u, guidance_scale)


def step(model, process, x_t, t, t_next, cond, guidance_scale=1.0, x_next=None, generator=None) -> Transition:
    pred = predict(model, x_t, t, cond, guidance_scale, process)
    return process.transition(pred, x_t, t, t_next, x_next=x_next, generator=generator)


@torch.no_grad()
def sample(model, process: DiffusionProcess, shape, cond: dict | None = None, num_steps: int = 20,
           guidance_scale: float = 1.0, generator=None, device=None, keep_trajectory: bool = False):
    """Run the reverse process from t=1 to t=0. Returns the final sample, or a ``Trajectory``."""
    device = device or next(model.parameters()).device
    cond = {k: v.to(device) for k, v in (cond or {}).items()}
    ts = process.timesteps(num_steps, device)
    x = process.prior(shape, cond, device, generator)
    xs, log_probs, stochastic = [x], [], []
    for k in range(num_steps):
        tr = step(model, process, x, ts[k], ts[k + 1], cond, guidance_scale, generator=generator)
        x = tr.x_next
        if keep_trajectory:
            xs.append(x)
            stochastic.append(tr.log_prob is not None)
            log_probs.append(tr.log_prob if tr.log_prob is not None else torch.zeros(shape[0], device=device))
    if not keep_trajectory:
        return x
    return Trajectory(
        torch.stack(xs, dim=1),
        torch.stack(log_probs, dim=1).float(),
        torch.tensor(stochastic, dtype=torch.bool),
        ts,
    )
