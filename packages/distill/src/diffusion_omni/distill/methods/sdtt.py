from __future__ import annotations

import copy

import torch
from diffusion_omni.core.registry import DISTILLERS

from .base import Distiller


@DISTILLERS.register("sdtt")
class SelfDistillationThroughTime(Distiller):
    """Self-distillation through time for masked diffusion LMs (Deschenaux & Gulcehre, 2024).

    In each round the student learns to do in one step what the round's teacher does in
    ``sub_steps`` steps. Then the student becomes the next round's teacher, halving the
    step count again. For a masked position, the target is the teacher's predictive
    distribution at the sub-step where the teacher decoded it, so it conditions on
    whatever the teacher decoded earlier inside the interval. That dependency is
    exactly what a single parallel step otherwise misses. Positions the teacher leaves
    masked take its last sub-step's distribution. Loss: KL(target || student) over
    masked positions.
    """

    processes = ("masked",)

    def __init__(self, start_steps: int = 32, rounds: int = 3, sub_steps: int = 2):
        if start_steps % sub_steps**rounds:
            raise ValueError(f"start_steps {start_steps} must be divisible by sub_steps**rounds")
        self.start_steps, self.rounds, self.sub_steps = start_steps, rounds, sub_steps
        self.round = 0

    def default_grid(self, tr):
        return torch.linspace(1.0, 0.0, self.start_steps // self.sub_steps**self.rounds + 1, device=tr.device)

    def setup(self, tr):
        self.round_teacher = copy.deepcopy(tr.teacher).requires_grad_(False).eval()
        self.round_len = max(1, tr.dc.steps // self.rounds)

    def state_dict(self):
        return {"round": self.round, "round_teacher": self.round_teacher.state_dict()}

    def load_state_dict(self, state, tr=None):
        self.round = int(state["round"])
        self.round_teacher.load_state_dict(state["round_teacher"])

    def schedule(self, it, tr):
        r = min(it // self.round_len, self.rounds - 1)
        if r != self.round:
            source = tr.roles["student"].ema
            with (source.swapped_in(tr.student) if source else torch.no_grad()):
                self.round_teacher.load_state_dict(tr.student.state_dict())
            self.round = r
        return ["student"]

    @torch.no_grad()
    def targets(self, tr, x_t, t, s, cond):
        proc, mask = tr.process, tr.process.mask_id
        ts = torch.linspace(float(t), float(s), self.sub_steps + 1, device=tr.device)
        target = None
        assigned = torch.zeros_like(x_t, dtype=torch.bool)
        x = x_t
        for j in range(self.sub_steps):
            logits = self.round_teacher(x, ts[j], cond)
            probs = proc.log_probs(logits).exp()
            target = probs.clone() if target is None else target
            step = proc.transition(logits, x, ts[j], ts[j + 1], generator=tr.gen)
            newly = (x == mask) & (step.x_next != mask)
            target[newly] = probs[newly]
            assigned |= newly
            x = step.x_next
        left = (x_t == mask) & ~assigned
        target[left] = probs[left]
        return target

    def loss(self, role, tr):
        b = tr.data.batch(tr.dc.batch_size)
        steps = self.start_steps // self.sub_steps ** (self.round + 1)
        grid = torch.linspace(1.0, 0.0, steps + 1, device=tr.device)
        k = int(tr.rng.integers(0, steps))
        t, s = grid[k], grid[k + 1]
        prefix = b.cond.get("prefix")
        x_t = tr.process.q_sample(b.x0, t, maskable=None if prefix is None else prefix < 0)
        target = self.targets(tr, x_t, t, s, b.cond)
        logp = tr.process.log_probs(tr.student(x_t, t, b.cond))
        masked = x_t == tr.process.mask_id
        kl = (torch.xlogy(target, target) - target * logp.masked_fill(~logp.isfinite(), 0.0)).sum(-1)
        loss = (kl * masked).sum() / masked.sum().clamp(min=1)
        return loss, {"sdtt_kl": loss.item(), "round": float(self.round), "student_steps": float(steps)}
