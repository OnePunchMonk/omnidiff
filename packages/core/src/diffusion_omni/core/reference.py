"""Reference samplers: slow, exact ground truth that faster samplers are checked against.

A reference sampler ignores the process's configured ``sampler`` and runs the process's exact
reverse dynamics at high NFE:

  continuous (``gaussian``, ``flow``): the probability-flow ODE with Heun's method. Heun here is
  the average of the model outputs at both ends of an ``ode_step``, which is exact Heun in each
  process's natural coordinate (velocity in t for flow, epsilon in sigma/alpha for VP). The last
  step, into t = 0, is Euler, as in EDM. On flow the error falls as 1/N^2; on VP the final Euler step
  limits it to about 1/N (~1e-3 at 500 steps on a Gaussian toy).
  ``masked``: ancestral sampling. With more steps than positions it approaches decoding one
  token at a time, which samples the model's joint exactly.

``dtype=torch.float64`` runs the model and the state in double precision on a copy of the model. The
processes still compute times and schedule coefficients in float32.
Few-step students, cached or quantised samplers and exported bundles are compared against the
output with ``parity``, starting both from the same prior.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from .processes.base import ContinuousProcess, DiffusionProcess
from .sampling import predict, resolve_timesteps


@torch.no_grad()
def reference_sample(model, process: DiffusionProcess, shape, cond: dict | None = None, num_steps: int = 500,
                     guidance_scale: float = 1.0, generator=None, device=None, dtype=torch.float64,
                     x_init: torch.Tensor | None = None, timesteps=None) -> torch.Tensor:
    """Sample with the process's exact reverse dynamics at ``num_steps`` steps (Heun for continuous processes).

    ``x_init`` replaces the prior draw, so a candidate sampler and the reference can share a start.
    ``dtype`` applies to continuous states and the model; masked states stay integer token ids.
    """
    device = device or next(model.parameters()).device
    if dtype is not None and dtype != next(model.parameters()).dtype:
        model = copy.deepcopy(model).to(dtype)
    cond = {k: v.to(device) for k, v in (cond or {}).items()}
    if getattr(process, "sampler", None) == "coupled":  # reference is the ancestral process, not its coupling
        process = copy.copy(process)
        process.sampler = "ancestral"
    ts = resolve_timesteps(process, num_steps, timesteps, device)
    x = process.prior(shape, cond, device, generator) if x_init is None else x_init.to(device)
    if not isinstance(process, ContinuousProcess):
        for k in range(len(ts) - 1):
            pred = predict(model, x, ts[k], cond, guidance_scale, process, s=ts[k + 1])
            x = process.transition(pred, x, ts[k], ts[k + 1], generator=generator).x_next
        return x
    if dtype is not None:
        x = x.to(dtype)
    for k in range(len(ts) - 1):
        t, t_next = ts[k], ts[k + 1]
        pred = predict(model, x, t, cond, guidance_scale, process, s=t_next)
        if float(t_next) > 0:
            x_euler = process.ode_step(pred, x, t, t_next)
            pred = 0.5 * (pred + predict(model, x_euler, t_next, cond, guidance_scale, process, s=t_next))
        x = process.ode_step(pred, x, t, t_next)
    return x


@dataclass
class Parity:
    max_abs: float   # largest absolute difference over every element
    rmse: float      # root mean squared difference
    match: float     # fraction of equal elements (meaningful for token ids)
    passed: bool     # max_abs <= atol for continuous samples, match >= min_match for tokens

    def as_dict(self) -> dict:
        return {"max_abs": self.max_abs, "rmse": self.rmse, "match": self.match, "passed": self.passed}


def parity(candidate: torch.Tensor, reference: torch.Tensor, atol: float = 1e-2, min_match: float = 1.0) -> Parity:
    """Elementwise agreement between a candidate sampler's output and the reference from the same start."""
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch: candidate {tuple(candidate.shape)} vs reference {tuple(reference.shape)}")
    a, b = candidate.detach().cpu(), reference.detach().cpu()
    match = (a == b).double().mean().item()
    diff = (a.double() - b.double()).abs()
    max_abs, rmse = diff.max().item(), diff.pow(2).mean().sqrt().item()
    passed = match >= min_match if not a.is_floating_point() else max_abs <= atol
    return Parity(max_abs, rmse, match, passed)
