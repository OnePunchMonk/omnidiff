from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusion_omni.core.registry import DISTILLERS

from .base import Distiller


@DISTILLERS.register("kd")
class TrajectoryKD(Distiller):
    """Knowledge distillation on teacher trajectories (Luhman & Luhman, 2021; the regression part of
    progressive distillation, Salimans & Ho, 2022).

    Teacher ODE states on the student's grid are regressed onto the trajectory's endpoint, in x0
    space for continuous processes and as cross-entropy on the teacher's final tokens for masked
    diffusion. It is the baseline the distribution-level methods must beat: regressing to a
    conditional mean blurs as steps shrink.
    """

    sampler = "ode"

    def setup(self, tr):
        if tr.dc.data != "teacher":
            raise ValueError("kd needs teacher trajectories: set distill.data: teacher")

    def loss(self, role, tr):
        b = tr.data.batch(tr.dc.batch_size)
        B, K = b.x0.shape[0], len(tr.grid) - 1
        k = torch.randint(0, K, (B,), device=tr.device)
        x_t, t = b.states[torch.arange(B, device=tr.device), k], tr.grid[k]
        pred = tr.student_pred(x_t, t, b.cond)
        if not tr.continuous:
            masked = x_t == tr.process.mask_id
            nll = F.nll_loss(tr.process.log_probs(pred).transpose(1, 2), b.x0, reduction="none")
            loss = (nll * masked).sum() / masked.sum().clamp(min=1)
            return loss, {"kd_nll": loss.item()}
        loss = F.mse_loss(tr.process.to_x0(pred, x_t, t), b.x0)
        return loss, {"kd_mse": loss.item()}
