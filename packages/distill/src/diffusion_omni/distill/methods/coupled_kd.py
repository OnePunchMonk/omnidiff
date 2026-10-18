from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusion_omni.core.registry import DISTILLERS

from .base import Distiller


@DISTILLERS.register("coupled_kd")
class CoupledKD(Distiller):
    """Noise-conditioned distillation for masked diffusion.

    With the ``coupled`` sampler, the teacher's K-step ancestral sampling is a deterministic function of
    per-trajectory noise (an unmask time per position and Gumbel noise per position and token), and
    its output law is unchanged. The student gets the same noise as input (``cond["token_noise"]``
    through a zero-initialised projection), decodes the same positions on a coarser grid, and is
    trained with cross-entropy to output the teacher's tokens for that noise. It is Luhman-style
    noise -> sample distillation carried over to discrete diffusion.

    Why it can beat per-position methods such as SDTT and plain KD: few-step masked sampling fails
    because tokens decoded together are drawn independently. A per-position target, however good,
    leaves that independence in place. Here the student sees every position's noise through
    attention, so it can reproduce the teacher's joint sample. Diversity comes from the noise, not
    from sampling the student's distribution, which is why it samples with ``coupled_gumbel: 0``.

    ``noise: gumbel`` feeds the student the (L, V) Gumbel noise, which only fits small vocabularies.
    ``noise: uniform`` couples through one inverse-CDF uniform per position instead, and the student
    reads it as ``noise_features`` Fourier features: O(L) noise for any vocabulary size.
    """

    processes = ("masked",)
    discrete_sampler = "coupled"
    teacher_sampler = "coupled"
    new_param_prefixes = ("s_embed.", "adapter.noise_proj.")

    def __init__(self, noise: str = "gumbel", noise_features: int = 64):
        if noise not in ("gumbel", "uniform"):
            raise ValueError(f"noise must be gumbel|uniform, got {noise!r}")
        self.noise, self.noise_features = noise, noise_features
        self.teacher_process_defaults = {"coupled_gumbel": 1.0, "coupled_noise": noise}
        self.student_process_defaults = {"coupled_gumbel": 0.0, "coupled_noise": noise}

    def student_spec(self, spec, tr):
        adapter = spec["adapter"] if isinstance(spec["adapter"], dict) else {"type": spec["adapter"]}
        dim = tr.c.context["vocab_size"] if self.noise == "gumbel" else self.noise_features
        return {**spec, "adapter": {**adapter, "noise_dim": dim}}

    def setup(self, tr):
        if tr.dc.data != "teacher":
            raise ValueError("coupled_kd distils the teacher's own noise -> sample map: set distill.data: teacher")

    def loss(self, role, tr):
        b = tr.data.batch(tr.dc.batch_size)
        B, K, mask = b.x0.shape[0], len(tr.grid) - 1, tr.process.mask_id
        k = torch.randint(0, K, (B,), device=tr.device)
        t = tr.grid[k]
        # the coupled teacher's state at t: positions whose unmask time is >= t are already decoded
        free = b.cond["prefix"] < 0 if "prefix" in b.cond else torch.ones_like(b.x0, dtype=torch.bool)
        x_t = torch.where(free & (b.cond["unmask_time"] < t[:, None]), mask, b.x0)
        logp = tr.process.log_probs(tr.student_pred(x_t, t, b.cond))
        masked = x_t == mask
        nll = F.nll_loss(logp.transpose(1, 2), b.x0, reduction="none")
        loss = (nll * masked).sum() / masked.sum().clamp(min=1)
        acc = ((logp.argmax(-1) == b.x0) & masked).sum() / masked.sum().clamp(min=1)
        return loss, {"ckd_nll": loss.item(), "ckd_token_acc": acc.item()}
