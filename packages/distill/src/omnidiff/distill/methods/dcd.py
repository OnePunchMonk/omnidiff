from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from omnidiff.core.registry import DISTILLERS

from .base import Distiller


@DISTILLERS.register("dcd")
class DiscreteConsistency(Distiller):
    """Multi-step consistency distillation for masked diffusion (discrete analogue of CD).

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

    This follows 2607.19686 (abstract only) in using a purely discrete consistency objective. Its
    multi-mask forward process and shared-Gumbel coupling are not implemented: the teacher here has a
    single mask state. Data is the SFT dataset (``data: text_lines``) or the teacher pool.
    """

    processes = ("masked",)

    def __init__(self, sub_steps: int = 2, boundary_prob: float = 0.25, target_decay: float = 0.95,
                 t_min: float = 0.05):
        self.sub_steps, self.boundary_prob, self.target_decay, self.t_min = sub_steps, boundary_prob, target_decay, t_min

    def setup(self, tr):
        self.target = copy.deepcopy(tr.teacher).requires_grad_(False).eval()

    def after_update(self, role, tr):
        with torch.no_grad():
            for p, q in zip(self.target.parameters(), tr.student.parameters()):
                p.lerp_(q, 1 - self.target_decay)

    @torch.no_grad()
    def targets(self, tr, x_t, t, s, cond):
        """Consistency target (B, L, V) for ``x_t`` at ``t``, via teacher ancestral steps to ``s`` (both (B,))."""
        mask = tr.process.mask_id
        x = x_t
        for j in range(self.sub_steps):
            a, b = t + (s - t) * j / self.sub_steps, t + (s - t) * (j + 1) / self.sub_steps
            logits = tr.teacher(x, a, cond)
            x = tr.teacher_process.transition(logits, x, a[:, None], b[:, None], generator=tr.gen).x_next
        probs = tr.process.log_probs(self.target(x, s, cond)).exp()
        decoded = (x_t == mask) & (x != mask)
        onehot = F.one_hot(torch.where(decoded, x, 0), probs.shape[-1]).to(probs.dtype)
        return torch.where(decoded[..., None], onehot, probs)

    @staticmethod
    def kl(target, logp):
        return (torch.xlogy(target, target) - target * logp.masked_fill(~logp.isfinite(), 0.0)).sum(-1)

    def loss(self, role, tr):
        b = tr.data.batch(tr.dc.batch_size)
        B, mask = b.x0.shape[0], tr.process.mask_id
        rand = lambda: torch.rand(B, device=tr.device, generator=tr.gen)  # noqa: E731
        t = self.t_min + (1 - self.t_min) * rand()
        s = torch.where(rand() < self.boundary_prob, torch.zeros_like(t), t * rand())
        # only free positions are masked: the pretraining loss mask if the data has one, else non-prefix positions
        maskable = b.cond.get("loss_mask")
        prefix = b.cond.get("prefix")
        maskable = maskable.bool() if maskable is not None else None if prefix is None else prefix < 0
        x_t = tr.process.q_sample(b.x0, t, generator=tr.gen, maskable=maskable)
        target = self.targets(tr, x_t, t, s, b.cond)
        logp = tr.process.log_probs(tr.student(x_t, t, b.cond))
        masked = x_t == mask
        loss = (self.kl(target, logp) * masked).sum() / masked.sum().clamp(min=1)
        return loss, {"dcd_kl": loss.item(), "mean_gap": (t - s).mean().item()}
