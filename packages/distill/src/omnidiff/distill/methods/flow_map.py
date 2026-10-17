from __future__ import annotations

import torch
from torch.func import jvp

from omnidiff.core.nn.layers import math_attention
from omnidiff.core.processes.base import bcast
from omnidiff.core.registry import DISTILLERS

from .base import Distiller


@DISTILLERS.register("flow_map")
class FlowMapDistillation(Distiller):
    """Flow-map distillation with the MeanFlow identity (Geng et al., 2025), the teacher supplying the velocity.

    The student is a two-time model u(x_t, t, s): the average velocity over [s, t], so that the
    flow map is X(x_t, t, s) = x_t + (s - t) u. Differentiating (t - s) u = int_s^t v along
    the teacher's ODE gives the training target

        u_tgt = v(x_t, t) - (t - s) * d/dt u(x_t, t, s),   d/dt = v . grad_x + partial_t,

    where d/dt u is one forward-mode JVP (or a finite difference) under stop-gradient. With s = t the
    target is the teacher's velocity. With ``s_mode: zero`` the student is a continuous-time consistency
    model, and it is exported with the renoise sampler.
    One student serves any step count. The zero-initialised s-embedding means it starts as the teacher's
    Euler sampler.
    """

    processes = ("flow",)
    sampler = "ode"
    two_time = True

    def __init__(self, equal_ratio: float = 0.25, s_mode: str = "uniform", derivative: str = "jvp",
                 fd_eps: float = 1e-3, time_sampling: str = "uniform", adaptive_p: float = 1.0,
                 adaptive_c: float = 1e-3):
        if s_mode not in ("uniform", "zero") or derivative not in ("jvp", "finite_difference"):
            raise ValueError("s_mode must be uniform|zero and derivative jvp|finite_difference")
        self.equal_ratio, self.s_mode, self.derivative, self.fd_eps = equal_ratio, s_mode, derivative, fd_eps
        self.time_sampling, self.adaptive_p, self.adaptive_c = time_sampling, adaptive_p, adaptive_c
        # s = 0 only: a consistency model, sampled by jumping to x0 and re-noising
        self.sampler = "renoise" if s_mode == "zero" else "ode"

    def sample_times(self, B: int, tr):
        if self.time_sampling == "uniform":
            t = torch.rand(B, device=tr.device)
        else:
            t = torch.sigmoid(torch.randn(B, device=tr.device))
        t = t.clamp(1e-3, 1.0)
        s = torch.zeros_like(t) if self.s_mode == "zero" else t * torch.rand(B, device=tr.device)
        equal = torch.rand(B, device=tr.device) < self.equal_ratio
        return t, torch.where(equal, t, s)

    def average_velocity(self, tr, z, t, s, v, cond):
        """u(z, t, s) with gradients, and its total time derivative along v without them."""
        fn = lambda z_, t_: tr.student(z_, t_, cond, s=s)  # noqa: E731
        if self.derivative == "jvp":
            with math_attention():
                u, dudt = jvp(fn, (z, t), (v, torch.ones_like(t)))
            return u, dudt.detach()
        u = fn(z, t)
        with torch.no_grad():
            h = self.fd_eps
            dudt = (u - fn(z - h * v, t - h)) / h  # backward difference keeps t - h inside (0, 1]
        return u, dudt

    def loss(self, role, tr):
        b = tr.data.batch(tr.dc.batch_size)
        B = b.x0.shape[0]
        t, s = self.sample_times(B, tr)
        z = tr.process.q_sample(b.x0, t, torch.randn_like(b.x0))
        v = tr.teacher_pred(z, t, b.cond)
        u, dudt = self.average_velocity(tr, z, t, s, v, b.cond)
        target = (v - bcast(t - s, z) * dudt).detach()
        err = ((u - target) ** 2).flatten(1).mean(1)
        w = 1.0 / (err.detach() + self.adaptive_c) ** self.adaptive_p
        loss = (w * err).mean()
        return loss, {"fm_mse": err.mean().item(), "fm_gap": (t - s).mean().item()}
