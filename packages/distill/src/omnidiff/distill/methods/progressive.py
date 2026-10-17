from __future__ import annotations

import copy

import torch

from omnidiff.core.processes.base import bcast
from omnidiff.core.registry import DISTILLERS
from omnidiff.core.sampling import predict

from .base import Distiller


@DISTILLERS.register("progressive")
class ProgressiveDistillation(Distiller):
    """Progressive distillation (Salimans & Ho, 2022) for continuous processes.

    Training runs in rounds. In round r the student, on a grid of N_r = start_steps / 2**(r+1)
    steps, learns to do in one deterministic ODE step what the round's teacher does in two steps
    of the 2 N_r grid. Round 0's teacher is the real teacher (with its guidance). Later rounds use
    a frozen snapshot of the previous round's EMA student, so the step count halves every round and
    the exported grid is the last round's.

    The target is derived from ``alpha_sigma`` alone, so it holds for any ContinuousProcess. Both
    ``x_t`` and the step's endpoint ``x_s`` are combinations of the same (x0, eps):

        x_t = a_t x0 + sigma_t eps,   x_s = a_s x0 + sigma_s eps.

    Solving for the pair that makes one ODE step land on the two-step result x_s* gives

        eps* = (a_s x_t - a_t x_s*) / (a_s sigma_t - a_t sigma_s),
        x0*  = (sigma_t x_s* - sigma_s x_t) / (sigma_t a_s - sigma_s a_t).

    The loss is taken in x0 space (the paper's choice: an eps-space error is amplified by
    1 / sigma_t near the data), weighted by max(SNR, 1) (the paper's truncated SNR weight).
    ``target_pred`` also returns the network output that realises (x0*, eps*) in the process's
    own parameterisation (eps for gaussian, velocity for flow).
    """

    processes = ("gaussian", "flow")
    sampler = "ode"

    def __init__(self, start_steps: int = 16, rounds: int = 3, weighting: str = "truncated_snr"):
        if start_steps % 2**rounds:
            raise ValueError(f"start_steps {start_steps} must be divisible by 2**rounds")
        if weighting not in ("truncated_snr", "none"):
            raise ValueError("weighting must be truncated_snr|none")
        self.start_steps, self.rounds, self.weighting = start_steps, rounds, weighting
        self.round = 0

    def grid(self, tr, r: int) -> torch.Tensor:
        """The student's grid in round ``r``. The round's teacher grid has twice the steps and contains this one."""
        return tr.process.timesteps(self.start_steps // 2 ** (r + 1), tr.device)

    def default_grid(self, tr):
        return self.grid(tr, self.rounds - 1)

    def setup(self, tr):
        self.round_teacher = None  # round 0 distils the real teacher
        self.round_len = max(1, tr.dc.steps // self.rounds)

    def schedule(self, it, tr):
        r = min(it // self.round_len, self.rounds - 1)
        if r != self.round:
            source = tr.roles["student"].ema
            if self.round_teacher is None:
                self.round_teacher = copy.deepcopy(tr.student).requires_grad_(False).eval()
            with (source.swapped_in(tr.student) if source else torch.no_grad()):
                self.round_teacher.load_state_dict(tr.student.state_dict())
            self.round = r
        return ["student"]

    def _teacher_pred(self, tr, x, t, cond):
        if self.round_teacher is None:
            return tr.teacher_pred(x, t, cond)
        with torch.no_grad():  # the snapshot is a student: guidance is already distilled in
            return predict(self.round_teacher, x, t, cond, 1.0, tr.process)

    @torch.no_grad()
    def two_teacher_steps(self, tr, x_t, t, mid, s, cond):
        p = tr.process
        x_mid = p.ode_step(self._teacher_pred(tr, x_t, t, cond), x_t, t, mid)
        return p.ode_step(self._teacher_pred(tr, x_mid, mid, cond), x_mid, mid, s)

    @staticmethod
    def x0_eps_target(process, x_t, x_s, t, s):
        """The (x0, eps) for which one ODE step from x_t at t lands on x_s at s."""
        a_t, sg_t = (bcast(v, x_t) for v in process.alpha_sigma(torch.as_tensor(t, device=x_t.device)))
        a_s, sg_s = (bcast(v, x_t) for v in process.alpha_sigma(torch.as_tensor(s, device=x_t.device)))
        eps = (a_s * x_t - a_t * x_s) / (a_s * sg_t - a_t * sg_s)
        x0 = (sg_t * x_s - sg_s * x_t) / (sg_t * a_s - sg_s * a_t)
        return x0, eps

    @staticmethod
    def target_pred(process, x_t, x_s, t, s):
        """The network output (in the process's parameterisation) whose ODE step from x_t at t lands on x_s."""
        _, eps = ProgressiveDistillation.x0_eps_target(process, x_t, x_s, t, s)
        zero = torch.zeros_like(x_t)
        e0, e1 = process.to_eps(zero, x_t, t), process.to_eps(zero + 1, x_t, t)
        return (eps - e0) / (e1 - e0)  # to_eps is affine in pred, with a per-time slope

    def loss(self, role, tr):
        b = tr.data.batch(tr.dc.batch_size)
        p = tr.process
        student_grid = self.grid(tr, self.round)
        teacher_grid = p.timesteps(2 * (len(student_grid) - 1), tr.device)
        k = torch.randint(0, len(student_grid) - 1, (b.x0.shape[0],), device=tr.device)
        t, s, mid = student_grid[k], student_grid[k + 1], teacher_grid[2 * k + 1]
        x_t = p.q_sample(b.x0, t, torch.randn_like(b.x0))
        x_s = self.two_teacher_steps(tr, x_t, t, mid, s, b.cond)
        x0_star, _ = self.x0_eps_target(p, x_t, x_s, t, s)
        err = ((p.to_x0(tr.student_pred(x_t, t, b.cond), x_t, t) - x0_star) ** 2).flatten(1).mean(1)
        if self.weighting == "truncated_snr":
            a, sg = p.alpha_sigma(t)
            w = (a**2 / sg**2).clamp(min=1.0)
            err = w / w.mean() * err
        loss = err.mean()
        return loss, {"progressive_x0_mse": loss.item(), "round": float(self.round),
                      "student_steps": float(len(student_grid) - 1)}
