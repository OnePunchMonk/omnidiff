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


def _twice(v):
    return v if v.dim() == 0 else torch.cat([v, v])


def predict(model, x, t, cond, guidance_scale: float = 1.0, process: DiffusionProcess | None = None, s=None):
    """Model output at (x, t), with classifier-free guidance when the model supports it.

    ``t`` (and ``s``) may be scalars or per-sample (B,). ``s`` is passed only to two-time models.
    """
    cond = cond or {}
    t = torch.as_tensor(t, device=x.device, dtype=torch.float32)
    kw = {}
    if s is not None and getattr(model, "two_time", False):
        kw["s"] = torch.as_tensor(s, device=x.device, dtype=torch.float32)
    if guidance_scale == 1.0 or not getattr(model, "supports_guidance", False):
        return model(x, t, cond, **kw)
    B = x.shape[0]
    both = {k: torch.cat([v, v]) for k, v in cond.items()}
    drop = torch.cat([torch.zeros(B, dtype=torch.bool), torch.ones(B, dtype=torch.bool)]).to(x.device)
    kw = {k: _twice(v) for k, v in kw.items()}
    c, u = model(torch.cat([x, x]), _twice(t), both, drop, **kw).chunk(2)
    return process.guide(c, u, guidance_scale)


def step(model, process, x_t, t, t_next, cond, guidance_scale=1.0, x_next=None, generator=None) -> Transition:
    # A two-time model is asked for the jump it takes: to t_next, or straight to data (s = 0) when
    # the sampler predicts x0 and re-noises.
    sampler = getattr(process, "sampler", None)
    s = torch.zeros_like(torch.as_tensor(t_next)) if sampler == "renoise" else t_next
    pred = predict(model, x_t, t, cond, guidance_scale, process, s=s)
    kw = {"cond": cond} if sampler == "coupled" else {}
    return process.transition(pred, x_t, t, t_next, x_next=x_next, generator=generator, **kw)


def resolve_timesteps(process: DiffusionProcess, num_steps: int, timesteps, device) -> torch.Tensor:
    """An explicit descending grid (e.g. a few-step student's) or the process's default grid."""
    if timesteps is None:
        return process.timesteps(num_steps, device)
    ts = torch.as_tensor(timesteps, dtype=torch.float32, device=device)
    if ts.dim() != 1 or len(ts) < 2 or not bool((ts[1:] < ts[:-1]).all()):
        raise ValueError(f"timesteps must be a strictly descending list, got {timesteps}")
    return ts


@torch.no_grad()
def sample(model, process: DiffusionProcess, shape, cond: dict | None = None, num_steps: int = 20,
           guidance_scale: float = 1.0, generator=None, device=None, keep_trajectory: bool = False,
           timesteps=None):
    """Run the reverse process over ``timesteps`` (default: the process grid with ``num_steps``).

    Returns the final sample, or a ``Trajectory``.
    """
    device = device or next(model.parameters()).device
    cond = {k: v.to(device) for k, v in (cond or {}).items()}
    ts = resolve_timesteps(process, num_steps, timesteps, device)
    num_steps = len(ts) - 1
    if getattr(process, "sampler", None) == "coupled":  # noise already in cond (e.g. a teacher pool) is kept
        cond = {**process.init_noise(shape, device, generator), **cond}
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
