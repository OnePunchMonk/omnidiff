from __future__ import annotations

from diffusion_omni.core.registry import DISTILLERS

from .base import Distiller
from .dmd2 import DMD2
from .flow_map import FlowMapDistillation


@DISTILLERS.register("rcm")
class ScoreRegularisedFlowMap(Distiller):
    """Flow-map / consistency distillation regularised with DMD2's reverse divergence, after rCM (Zheng et al., 2026).

    rCM adds a DMD score-distillation term to continuous-time consistency (sCM), because sCM's
    forward-divergence objective loses fine detail and the mode-seeking DMD term restores it.
    Here the consistency side is the flow-map objective in the rectified-flow parameterisation.
    ``s_mode: zero`` is the continuous-time consistency model, which matches rCM's sCM + DMD.
    ``s_mode: uniform`` regularises a full flow map instead. Student loss = flow-map loss +
    ``dmd_weight`` * DMD2 generator loss. The fake score trains exactly as in DMD2.
    """

    processes = ("flow",)
    sampler = "ode"
    two_time = True

    def __init__(self, dmd_weight: float = 0.1, fake_updates: int = 5, s_mode: str = "zero", equal_ratio: float = 0.25,
                 derivative: str = "jvp", tau_min: float = 0.02, tau_max: float = 0.98, fake_lr: float | None = None):
        self.flow_map = FlowMapDistillation(equal_ratio=equal_ratio, s_mode=s_mode, derivative=derivative)
        self.dmd = DMD2(fake_updates=fake_updates, tau_min=tau_min, tau_max=tau_max, fake_lr=fake_lr)
        self.dmd_weight = dmd_weight
        self.sampler = self.flow_map.sampler

    def setup(self, tr):
        self.dmd.setup(tr)

    def schedule(self, it, tr):
        return self.dmd.schedule(it, tr)

    def loss(self, role, tr):
        if role == "fake":
            return self.dmd.loss(role, tr)
        fm_loss, metrics = self.flow_map.loss(role, tr)
        dm_loss, dm_metrics = self.dmd.loss(role, tr)
        metrics.update(dm_metrics)
        return fm_loss + self.dmd_weight * dm_loss, metrics
