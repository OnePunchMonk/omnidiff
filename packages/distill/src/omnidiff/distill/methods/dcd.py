from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from omnidiff.core.registry import DISTILLERS

from .base import Distiller


@DISTILLERS.register("dcd")
class DiscreteConsistency(Distiller):
    """Single-mask discrete consistency distillation for masked diffusion (an ablation).

    A consistency function maps any state on a teacher trajectory to the same clean sample. For an
    absorbing process the trajectory's endpoint is stochastic, so the function is the *distribution*
    over clean tokens, f(x_t, t). Pairs on one trajectory come from ``sub_steps`` teacher ancestral
    steps x_t -> x_s. The student's prediction at (x_t, t) is regressed (KL) onto a stop-gradient
    target built from x_s:

    - a position the teacher decoded on the way takes a one-hot at its token, since f(x_s, s) is
      a point mass there; with s = 0 (probability ``boundary_prob``) that anchors every position to
      the teacher's sample, the boundary condition f(x_0, 0) = x_0;
    - a position still masked at s takes the target network's distribution at (x_s, s).

    The target network is an EMA of the student (``target_decay``), initialised from the teacher. The
    fixed point is a student that agrees with its target on every pair. ``s`` is drawn per example,
    so one network is trained for any step count.

    This is NOT the scheme of 2607.19686 (read from its abstract only). Its multi-mask forward process
    and shared-Gumbel coupling are not implemented: the teacher here has a single mask state, and the
    target is a factorised per-position distribution. A flat training KL is expected, since the
    sampled one-hot targets carry irreducible entropy, and so is a weak few-step result. Data is the SFT
    dataset (``data: text_lines``) or the teacher pool.
    """

    processes = ("masked",)

    def __init__(self, sub_steps: int = 2, boundary_prob: float = 0.25, target_decay: float = 0.95,
                 t_min: float = 0.05):
        self.sub_steps, self.boundary_prob, self.target_decay, self.t_min = sub_steps, boundary_prob, target_decay, t_min

    def setup(self, tr):
        # copied from the student, after LoRA is applied, so the two share parameter and buffer names
        self.target = copy.deepcopy(tr.student).requires_grad_(False).eval()

    def state_dict(self):
        return {"target": self.target.state_dict()}

    def load_state_dict(self, state, tr=None):
        target = state.get("target")
        if target is None:
            raise ValueError("dcd checkpoint has no target state")
        self.target.load_state_dict(target)

    def after_update(self, role, tr):
        if role != "student":
            return
        with torch.no_grad():
            state = tr.student.state_dict()
            for k, p in self.target.state_dict().items():
                if p.is_floating_point():
                    p.lerp_(state[k], 1 - self.target_decay)
                else:
                    p.copy_(state[k])

    @torch.no_grad()
    def targets(self, tr, x_t, t, s, cond):
        """Consistency target (B, L, V) for ``x_t`` at ``t``, via teacher ancestral steps to ``s`` (both (B,))."""
        mask = tr.process.mask_id
        x = x_t
        for j in range(self.sub_steps):
            a, b = t + (s - t) * j / self.sub_steps, t + (s - t) * (j + 1) / self.sub_steps
            logits = tr.teacher_pred(x, a, cond)
            x = tr.teacher_process.transition(logits, x, a[:, None], b[:, None], generator=tr.gen).x_next
        probs = tr.process.log_probs(self.target(x, s, cond)).exp()
        decoded = (x_t == mask) & (x != mask)
        onehot = F.one_hot(torch.where(decoded, x, 0), probs.shape[-1]).to(probs.dtype)
        return torch.where(decoded[..., None], onehot, probs)

    @staticmethod
    def kl(target, logp):
        return (torch.xlogy(target, target) - target * logp.masked_fill(~logp.isfinite(), 0.0)).sum(-1)

    @staticmethod
    def maskable(cond):
        """Positions that may be masked: inside the loss mask if there is one, and never a prefix."""
        loss_mask, prefix = cond.get("loss_mask"), cond.get("prefix")
        loss_mask = loss_mask.bool() if loss_mask is not None else None
        if prefix is None:
            return loss_mask
        return prefix < 0 if loss_mask is None else loss_mask & (prefix < 0)

    def loss(self, role, tr):
        b = tr.data.batch(tr.dc.batch_size)
        B, mask = b.x0.shape[0], tr.process.mask_id
        rand = lambda: torch.rand(B, device=tr.device, generator=tr.gen)  # noqa: E731
        t = self.t_min + (1 - self.t_min) * rand()
        s = torch.where(rand() < self.boundary_prob, torch.zeros_like(t), t * rand())
        x_t = tr.process.q_sample(b.x0, t, generator=tr.gen, maskable=self.maskable(b.cond))
        target = self.targets(tr, x_t, t, s, b.cond)
        logp = tr.process.log_probs(tr.student(x_t, t, b.cond))
        masked = x_t == mask
        n = masked.sum().clamp(min=1)
        loss = (self.kl(target, logp) * masked).sum() / n
        # the objective's fixed point is the teacher's one-step marginal, so KL(teacher || student) on
        # fresh states tells "at the fixed point" (near 0) from "not training" (large)
        with torch.no_grad():
            p = tr.process.log_probs(tr.teacher_pred(x_t, t, b.cond)).exp()
            to_teacher = (self.kl(p, logp.detach()) * masked).sum() / n
        return loss, {"dcd_kl": loss.item(), "kl_to_teacher": to_teacher.item(), "mean_gap": (t - s).mean().item()}
