from __future__ import annotations

import torch

from omnidiff.core.registry import DISTILLERS, REWARDS

from .dmd2 import DMD2


@DISTILLERS.register("dmd2_reward")
class DMD2Reward(DMD2):
    """DMD2 with a reward term in the generator loss.

    ``reward`` is a ``REWARDS`` spec. It is applied to the student's x0 in model space with the same
    transform as ``codec.decode`` (pixel: clamp to [-1, 1], map to [0, 1]), so training and evaluation
    see the same reward. At ``reward_weight = 0`` the generator loss and its gradient equal DMD2's for a given x0, and the
    prompts are drawn as in DMD2 (no grouping).

    ``mode: gradient`` (differentiable rewards, via the reward's ``torch_call``):

        loss = loss_dmd2 - reward_weight * mean_i r(x0_i)

    The reward gradient reaches the student through x0, at whichever stage the rollout was drawn (the
    reward applies to that stage's x0 prediction). The two terms are not on a common scale: the DMD2
    gradient on x0 is normalised per sample, the reward gradient is not, and in a probe the reward gradient
    at weight 1 was about 10x the DMD2 one. The metric ``reward_grad_ratio`` logs the ratio of mean
    gradient magnitudes on x0.

    ``mode: advantage`` (any reward, including non-differentiable ones): the per-sample DMD2 gradient is
    scaled by w_i = g * softmax_i(reward_weight * A_i) within each group of ``group_size`` (g) student
    rollouts that share a prompt, where A_i = (r_i - mean r) / (std r + eps) is the group-normalised,
    detached advantage (as in GRPO). Comparing samples of one prompt keeps the prompt's own reward
    level out of the advantage. This is reward-weighted regression on the
    distribution-matching update: samples the reward likes keep more of their pull toward the teacher
    distribution, samples it dislikes keep less. w has mean 1, so the step size is unchanged, and
    ``reward_weight = 0`` gives w = 1. The weights are applied as a gradient scale on x0
    (value unchanged up to rounding), which reuses ``DMD2.generator_loss`` as is. Unlike ``gradient``, this only
    reweights samples the student already produces; it does not move them along the reward.
    """

    def __init__(self, reward: dict, reward_weight: float = 1.0, mode: str = "gradient", group_size: int = 4, **kw):
        super().__init__(**kw)
        if mode not in ("gradient", "advantage"):
            raise ValueError(f"mode must be 'gradient' or 'advantage', not {mode!r}")
        if mode == "advantage" and group_size < 2:
            raise ValueError("advantage mode needs group_size >= 2")
        self.reward_spec, self.reward_weight, self.mode, self.group_size = reward, float(reward_weight), mode, group_size
        self.prompts: list | None = None  # the prompts behind the generator batch, for the reward

    def setup(self, tr):
        super().setup(tr)
        self.reward = REWARDS.build(self.reward_spec, **tr.c.context)
        if self.mode == "gradient" and not getattr(self.reward, "differentiable", False):
            raise ValueError(f"reward {type(self.reward).__name__} is not differentiable; use mode: advantage")

    def loss(self, role, tr):
        if role != "student":
            return super().loss(role, tr)
        n = tr.dc.batch_size
        # advantage mode: g consecutive samples share a prompt (skipped at weight 0, which is plain DMD2)
        g = self.group_size if self.mode == "advantage" and self.reward_weight != 0 else 1
        if n % g:
            raise ValueError(f"batch_size {n} is not a multiple of group_size {g}")
        saved = vars(tr).get("prompt_cond")

        def prompt_cond(m):  # same draws as Trainer.prompt_cond, but the prompts are kept for the reward
            prompts = tr.c.prompts.sample(m // g, tr.rng)
            self.prompts = [p for p in prompts for _ in range(g)]
            cond = tr.c.prompts.encode(prompts, tr.device)
            return {k: v.repeat_interleave(g, 0) for k, v in cond.items()}

        tr.prompt_cond = prompt_cond  # DMD2.generate and its rollout run unchanged on this cond
        try:
            return super().loss(role, tr)
        finally:
            if saved is None:
                del tr.prompt_cond
            else:
                tr.prompt_cond = saved

    def _prompts_for(self, n):
        return self.prompts if self.prompts is not None and len(self.prompts) == n else [""] * n

    def generator_loss(self, tr, x0_g, cond):
        n = x0_g.shape[0]
        prompts = self._prompts_for(n)
        if self.mode == "advantage":
            r = torch.as_tensor(self.reward(tr.c.codec.decode(x0_g), prompts), dtype=x0_g.dtype, device=x0_g.device)
            g = self.group_size
            r_g = r.view(-1, g)
            adv = (r_g - r_g.mean(1, keepdim=True)) / (r_g.std(1, unbiased=False, keepdim=True) + 1e-6)
            w = (torch.softmax(self.reward_weight * adv, 1) * g).view(-1, *[1] * (x0_g.dim() - 1))
            x0_scaled = x0_g * w + x0_g.detach() * (1 - w)
            loss, metrics = super().generator_loss(tr, x0_scaled, cond)
            metrics.update(reward=r.mean().item(), adv_weight_max=w.max().item())
            return loss, metrics
        loss, metrics = super().generator_loss(tr, x0_g, cond)
        r = self.reward.torch_call(tr.c.codec.decode_tensor(x0_g), prompts)
        metrics.update(reward=r.mean().item())
        if x0_g.requires_grad:
            # reward pull vs DMD2 pull on x0: dm_grad / numel is the DMD2 gradient's mean magnitude
            (dr,) = torch.autograd.grad(r.mean(), x0_g, retain_graph=True, allow_unused=True)
            if dr is not None:
                metrics["reward_grad_ratio"] = (self.reward_weight * dr.abs().mean() / (metrics["dm_grad"] / x0_g.numel() + 1e-12)).item()
        return loss - self.reward_weight * r.mean(), metrics
