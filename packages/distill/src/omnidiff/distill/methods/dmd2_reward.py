from __future__ import annotations

import torch

from omnidiff.core.registry import DISTILLERS, REWARDS

from .dmd2 import DMD2


@DISTILLERS.register("dmd2_reward")
class DMD2Reward(DMD2):
    """DMD2 with a reward term in the generator loss.

    ``reward`` is a ``REWARDS`` spec. It is applied to the student's x0 in model space with the same
    transform as ``codec.decode`` (pixel: clamp to [-1, 1], map to [0, 1]), so training and evaluation
    see the same reward. ``reward_weight = 0`` is exactly DMD2.

    ``mode: gradient`` (differentiable rewards, via the reward's ``torch_call``):

        loss = loss_dmd2 - reward_weight * mean_i r(x0_i)

    The reward gradient reaches the student through x0. The DMD2 term is normalised per sample, so
    ``reward_weight`` is relative to a gradient of roughly unit mean magnitude.

    ``mode: advantage`` (any reward, including non-differentiable ones): the per-sample DMD2 gradient is
    scaled by w_i = g * softmax_i(reward_weight * A_i) within each group of ``group_size`` (g) student
    rollouts that share a prompt, where A_i = (r_i - mean r) / (std r + eps) is the group-normalised,
    detached advantage (as in GRPO). Comparing samples of one prompt keeps the prompt's own reward
    level out of the advantage. This is reward-weighted regression on the
    distribution-matching update: samples the reward likes keep more of their pull toward the teacher
    distribution, samples it dislikes keep less. w has mean 1, so the step size is unchanged, and
    ``reward_weight = 0`` gives w = 1. The weights are applied as a gradient scale on x0
    (value unchanged), which reuses ``DMD2.generator_loss`` as is. Unlike ``gradient``, this only
    reweights samples the student already produces; it does not move them along the reward.
    """

    def __init__(self, reward: dict, reward_weight: float = 1.0, mode: str = "gradient", group_size: int = 4, **kw):
        super().__init__(**kw)
        self.group_size, self._grouped = group_size, False
        if mode not in ("gradient", "advantage"):
            raise ValueError(f"mode must be 'gradient' or 'advantage', not {mode!r}")
        self.reward, self.reward_weight, self.mode = REWARDS.build(reward), float(reward_weight), mode
        if mode == "gradient" and not getattr(self.reward, "differentiable", False):
            raise ValueError(f"reward {type(self.reward).__name__} is not differentiable; use mode: advantage")

    def loss(self, role, tr):
        if role != "student" or self.mode != "advantage":
            return super().loss(role, tr)
        n, g = tr.dc.batch_size, self.group_size
        if n % g:
            raise ValueError(f"batch_size {n} is not a multiple of group_size {g}")
        # g consecutive samples share a prompt; DMD2.generate (and its rollout) runs unchanged on that cond
        prompt_cond = tr.prompt_cond
        tr.prompt_cond = lambda m: {k: v.repeat_interleave(g, 0) for k, v in prompt_cond(m // g).items()}
        try:
            return super().loss(role, tr)
        finally:
            del tr.prompt_cond

    def generator_loss(self, tr, x0_g, cond):
        n = x0_g.shape[0]
        if self.mode == "advantage":
            r = torch.as_tensor(self.reward(tr.c.codec.decode(x0_g), [""] * n), dtype=x0_g.dtype, device=x0_g.device)
            g = self.group_size
            r_g = r.view(-1, g)
            adv = (r_g - r_g.mean(1, keepdim=True)) / (r_g.std(1, unbiased=False, keepdim=True) + 1e-6)
            w = (torch.softmax(self.reward_weight * adv, 1) * g).view(-1, *[1] * (x0_g.dim() - 1))
            x0_scaled = x0_g * w + x0_g.detach() * (1 - w)
            loss, metrics = super().generator_loss(tr, x0_scaled, cond)
            metrics.update(reward=r.mean().item(), adv_weight_max=w.max().item())
            return loss, metrics
        loss, metrics = super().generator_loss(tr, x0_g, cond)
        r = self.reward.torch_call(tr.c.codec.decode_tensor(x0_g), [""] * n)
        metrics.update(reward=r.mean().item())
        return loss - self.reward_weight * r.mean(), metrics
