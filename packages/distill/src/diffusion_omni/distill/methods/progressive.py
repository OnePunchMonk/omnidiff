from __future__ import annotations

import copy

import torch
from diffusion_omni.core.processes.base import bcast
from diffusion_omni.core.registry import DISTILLERS
from diffusion_omni.core.sampling import predict

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
    1 / sigma_t near the data). ``weighting``: ``truncated_snr`` is the paper's max(SNR, 1), which
    suits flow. For an eps-prediction network an x0 error is (sigma/alpha)^2 times the eps error,
    ~2e4 at t = 1 for vp_linear, so that weight lets t = 1 dominate; ``snr`` weights by SNR, which
    makes the loss the plain eps-space MSE.
    ``target_pred`` also returns the network output that realises (x0*, eps*) in the process's
    own parameterisation (eps for gaussian, velocity for flow).
    """

    processes = ("gaussian", "flow")
    sampler = "ode"

    def __init__(self, start_steps: int = 16, rounds: int = 3, weighting: str = "truncated_snr"):
        if start_steps % 2**rounds:
            raise ValueError(f"start_steps {start_steps} must be divisible by 2**rounds")
        if weighting not in ("truncated_snr", "snr", "none"):
            raise ValueError("weighting must be truncated_snr|snr|none")
        self.start_steps, self.rounds, self.weighting = start_steps, rounds, weighting
        self.round = 0

    def grid(self, tr, r: int) -> torch.Tensor:
        """The student's grid in round ``r``. The round's teacher grid has twice the steps and contains this one."""
        return tr.process.timesteps(self.start_steps // 2 ** (r + 1), tr.device)

    def default_grid(self, tr):
        return self.grid(tr, 0)  # ``schedule`` moves ``tr.grid`` to each round's grid, so an export is the round's student

    def setup(self, tr):
        if tr.dc.steps < self.rounds:
            raise ValueError(f"progressive needs distill.steps >= rounds ({tr.dc.steps} < {self.rounds})")
        if tr.dc.student_steps is not None or tr.dc.student_timesteps is not None:
            raise ValueError("progressive sets the student grid from start_steps and rounds; "
                             "do not set student_steps / student_timesteps")
        self.round_teacher = None  # round 0 distils the real teacher
        self.round_len = tr.dc.steps // self.rounds

    def state_dict(self):
        return {
            "round": self.round,
            "round_teacher": None if self.round_teacher is None else self.round_teacher.state_dict(),
        }

    def load_state_dict(self, state, tr):
        self.round = int(state["round"])
        teacher = state.get("round_teacher")
        if teacher is None:
            self.round_teacher = None
            return
        if self.round_teacher is None:
            self.round_teacher = copy.deepcopy(tr.student).requires_grad_(False).eval()
        self.round_teacher.load_state_dict(teacher)

    def schedule(self, it, tr):
        r = min(it // self.round_len, self.rounds - 1)
        if r != self.round:
            source = tr.roles["student"].ema
            if self.round_teacher is None:
                self.round_teacher = copy.deepcopy(tr.student).requires_grad_(False).eval()
            with (source.swapped_in(tr.student) if source else torch.no_grad()):
                self.round_teacher.load_state_dict(tr.student.state_dict())
            # as in the paper, the next round's student restarts from the snapshot, with fresh optimiser state
            tr.student.load_state_dict(self.round_teacher.state_dict())
            role = tr.roles["student"]
            role.optimizer.state.clear()
            if source:
                source.shadow = {k: v.detach().clone() for k, v in tr.student.state_dict().items() if k in source.shadow}
            self.round = r
            tr.grid = self.grid(tr, r)
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
        x0, _ = ProgressiveDistillation.x0_eps_target(process, x_t, x_s, t, s)
        # to_x0 is affine in pred with a nonzero slope for t > 0 (to_eps has none at t = 1 for flow);
        # probe it without the gaussian x0 clip, which would break the affine map
        process = copy.copy(process)
        if hasattr(process, "clip_x0"):
            process.clip_x0 = None
        zero = torch.zeros_like(x_t)
        a0, a1 = process.to_x0(zero, x_t, t), process.to_x0(zero + 1, x_t, t)
        return (x0 - a0) / (a1 - a0)

    @staticmethod
    def sample_triple(tr, student_grid, teacher_grid, n):
        """Per-sample (t, mid, s): a student interval and the teacher-grid time halfway through it."""
        k = torch.randint(0, len(student_grid) - 1, (n,), device=tr.device)
        return student_grid[k], teacher_grid[2 * k + 1], student_grid[k + 1]

    def loss(self, role, tr):
        b = tr.data.batch(tr.dc.batch_size)
        p = tr.process
        student_grid = self.grid(tr, self.round)
        teacher_grid = p.timesteps(2 * (len(student_grid) - 1), tr.device)
        t, mid, s = self.sample_triple(tr, student_grid, teacher_grid, b.x0.shape[0])
        x_t = p.q_sample(b.x0, t, torch.randn_like(b.x0))
        x_s = self.two_teacher_steps(tr, x_t, t, mid, s, b.cond)
        x0_star, _ = self.x0_eps_target(p, x_t, x_s, t, s)
        err = ((p.to_x0(tr.student_pred(x_t, t, b.cond), x_t, t) - x0_star) ** 2).flatten(1).mean(1)
        if self.weighting == "truncated_snr":
            a, sg = p.alpha_sigma(t)
            w = (a**2 / sg**2).clamp(min=1.0)
            err = w / w.mean() * err
        elif self.weighting == "snr":
            a, sg = p.alpha_sigma(t)
            err = a**2 / sg**2 * err
        loss = err.mean()
        return loss, {"progressive_loss": loss.item(), "round": float(self.round),
                      "student_steps": float(len(student_grid) - 1)}
