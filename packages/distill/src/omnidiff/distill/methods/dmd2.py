from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnidiff.core.registry import DISTILLERS

from .base import Distiller, per_sample_mean


class DiscriminatorHead(nn.Module):
    """Classifies noisy samples as real or generated from the fake-score network's features (DMD2's GAN)."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(self, h):
        return self.net(h.mean(1)).squeeze(-1)


@DISTILLERS.register("dmd2")
class DMD2(Distiller):
    """Distribution Matching Distillation 2 (Yin et al., 2024).

    The student's samples x0 are pushed along (real score - fake score). The real score is the frozen
    teacher, with CFG. The fake score is a copy of the teacher trained online to denoise student
    samples. In x0-prediction form the generator gradient is

        grad = (x0_fake(x_tau) - x0_real(x_tau)) / mean|x0 - x0_real(x_tau)|,  x_tau = q(x0, tau),

    applied through ``0.5 * ||x0 - sg(x0 - grad)||^2``. Two time-scale updates
    (``fake_updates`` fake steps per student step), backward simulation (the student
    is trained on its own intermediate states, as at inference), and an optional GAN
    head on the fake score's features, with real data or teacher samples as "real".
    """

    processes = ("gaussian", "flow")
    sampler = "renoise"

    def __init__(self, fake_updates: int = 5, tau_min: float = 0.02, tau_max: float = 0.98,
                 gan_weight: float = 0.0, backward_sim: bool = True, fake_lr: float | None = None,
                 last_stage_prob: float | None = None):
        self.fake_updates, self.tau_min, self.tau_max = fake_updates, tau_min, tau_max
        self.gan_weight, self.backward_sim, self.fake_lr = gan_weight, backward_sim, fake_lr
        self.last_stage_prob = last_stage_prob

    def setup(self, tr):
        fake = copy.deepcopy(tr.teacher).requires_grad_(True).train()
        modules = {"score": fake}
        if self.gan_weight > 0:
            modules["head"] = DiscriminatorHead(tr.teacher_cfg["model"]["dim"]).to(tr.device)
        self.fake = nn.ModuleDict(modules)
        tr.add_role("fake", self.fake, lr=self.fake_lr)

    def schedule(self, it, tr):
        return ["fake"] * self.fake_updates + ["student"]

    def _stage(self, tr) -> int:
        """Grid index the generator starts from: uniform, or the last stage with ``last_stage_prob``.

        The DM gradient matches the *mixture* of the stages' outputs to the data, while inference
        returns only the last stage's, so uniform stages let the last one drift (see docs/notes/D1.md).
        """
        last = len(tr.grid) - 2
        if self.last_stage_prob is not None and last > 0 and tr.rng.random() < self.last_stage_prob:
            return last
        return int(tr.rng.integers(0, last + 1))

    def generate(self, tr, n: int):
        """Student samples x0 (with grad), from its own state at a random grid step.

        The rollout to grid[k] is ``no_grad``, as in the DMD2 paper: earlier steps only produce the input
        state (the student's own renoised output, the state inference meets at grid[k]); the gradient
        flows through the final x0 prediction at t = grid[k] alone.
        """
        cond = tr.prompt_cond(n)
        k = self._stage(tr) if self.backward_sim else 0
        x = tr.student_rollout(cond, k, n)
        t = tr.grid[k]
        # s = 0 asks a two-time (flow-map) student for the jump straight to data; one-time models ignore it
        return tr.student_process.to_x0(tr.student_pred(x, t, cond, s=torch.zeros(())), x, t), cond

    def _noisy(self, tr, x0):
        tau = torch.rand(x0.shape[0], device=x0.device) * (self.tau_max - self.tau_min) + self.tau_min
        return tr.process.q_sample(x0, tau, torch.randn_like(x0)), tau

    def loss(self, role, tr):
        n = tr.dc.batch_size
        if role == "fake":
            with torch.no_grad():
                x0_g, cond = self.generate(tr, n)
            loss = tr.process.training_loss(self.fake["score"], x0_g, cond)
            metrics = {"fake_denoise": loss.item()}
            if self.gan_weight > 0:
                real = tr.data.batch(n)
                x_r, tau_r = self._noisy(tr, real.x0)
                x_f, tau_f = self._noisy(tr, x0_g)
                d_real = self.fake["head"](self.fake["score"].features(x_r, tau_r, real.cond))
                d_fake = self.fake["head"](self.fake["score"].features(x_f, tau_f, cond))
                d_loss = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
                loss = loss + self.gan_weight * d_loss
                metrics.update(d_loss=d_loss.item(), d_real=torch.sigmoid(d_real).mean().item())
            return loss, metrics

        return self.generator_loss(tr, *self.generate(tr, n))

    def generator_loss(self, tr, x0_g, cond):
        x_tau, tau = self._noisy(tr, x0_g)
        with torch.no_grad():
            real = tr.process.to_x0(tr.teacher_pred(x_tau, tau, cond), x_tau, tau)
            fake = tr.process.to_x0(self.fake["score"](x_tau, tau, cond), x_tau, tau)
            weight = 1.0 / (per_sample_mean((x0_g - real).abs()) + 1e-6)
            grad = (fake - real) * weight.view(-1, *[1] * (x0_g.dim() - 1))
        loss = 0.5 * F.mse_loss(x0_g, (x0_g - grad).detach())
        metrics = {"dm_loss": loss.item(), "dm_grad": grad.abs().mean().item()}
        if self.gan_weight > 0:
            g_loss = F.softplus(-self.fake["head"](self.fake["score"].features(x_tau, tau, cond))).mean()
            loss = loss + self.gan_weight * g_loss
            metrics["g_loss"] = g_loss.item()
        return loss, metrics
