from __future__ import annotations

import copy
import math

import torch
from diffusion_omni.core.registry import DISTILLERS

from .base import Distiller, per_sample_mean


@DISTILLERS.register("consistency")
class ConsistencyDistillation(Distiller):
    """Consistency distillation (Song et al., 2023), discrete time, with iCT's choices (Song & Dhariwal, 2023).

    f(x, t) = to_x0(student(x, t)) already satisfies the boundary condition f(x, 0) = x for
    both continuous processes, so no skip scaling is needed. For adjacent grid times
    t_hi > t_lo, the teacher takes one ODE step x_hi -> x_lo and the student is trained so that
    f(x_hi, t_hi) = sg(f_target(x_lo, t_lo)). The target is the current student under
    stop-gradient (iCT, ``target: stopgrad``) or an EMA copy (original CD, ``target: ema``).
    Distance: pseudo-Huber with c = 0.00054 * sqrt(d) (iCT) or squared L2, weighted by
    1 / (t_hi - t_lo).
    """

    processes = ("gaussian", "flow")
    sampler = "renoise"

    def __init__(self, num_intervals: int = 32, target: str = "stopgrad", target_decay: float = 0.95,
                 distance: str = "pseudo_huber", weighting: str = "inverse_dt"):
        if target not in ("stopgrad", "ema") or distance not in ("pseudo_huber", "l2"):
            raise ValueError("target must be stopgrad|ema and distance pseudo_huber|l2")
        self.num_intervals, self.target, self.target_decay = num_intervals, target, target_decay
        self.distance, self.weighting = distance, weighting

    def setup(self, tr):
        self.target_model = copy.deepcopy(tr.student).requires_grad_(False) if self.target == "ema" else None
        self.c = 0.00054 * math.sqrt(math.prod(tr.shape))
        self.teacher_grid = tr.process.timesteps(self.num_intervals, tr.device)

    def state_dict(self):
        return {"target_model": None if self.target_model is None else self.target_model.state_dict()}

    def load_state_dict(self, state, tr=None):
        target = state.get("target_model")
        if target is not None:
            if self.target_model is None:
                raise ValueError("consistency checkpoint has an EMA target but the method does not")
            self.target_model.load_state_dict(target)
        elif self.target_model is not None:
            raise ValueError("consistency checkpoint has no target_model state but target='ema'")

    def after_update(self, role, tr):
        if self.target_model is not None:
            with torch.no_grad():
                for p_t, p in zip(self.target_model.parameters(), tr.student.parameters()):
                    p_t.lerp_(p, 1 - self.target_decay)

    def loss(self, role, tr):
        b = tr.data.batch(tr.dc.batch_size)
        B = b.x0.shape[0]
        n = torch.randint(0, self.num_intervals, (B,), device=tr.device)
        t_hi, t_lo = self.teacher_grid[n], self.teacher_grid[n + 1]
        x_hi = tr.process.q_sample(b.x0, t_hi, torch.randn_like(b.x0))
        with torch.no_grad():
            x_lo = tr.process.ode_step(tr.teacher_pred(x_hi, t_hi, b.cond), x_hi, t_hi, t_lo)
            target_model = self.target_model or tr.student
            target = tr.process.to_x0(tr.student_pred(x_lo, t_lo, b.cond, model=target_model), x_lo, t_lo)
        out = tr.process.to_x0(tr.student_pred(x_hi, t_hi, b.cond), x_hi, t_hi)
        sq = ((out - target) ** 2).flatten(1).sum(1)
        d = (sq + self.c**2).sqrt() - self.c if self.distance == "pseudo_huber" else sq / out[0].numel()
        w = 1.0 / (t_hi - t_lo) if self.weighting == "inverse_dt" else torch.ones_like(t_hi)
        loss = (w / w.mean() * d).mean()
        return loss, {"cd_distance": d.mean().item(), "cd_l2": per_sample_mean((out - target) ** 2).mean().item()}
