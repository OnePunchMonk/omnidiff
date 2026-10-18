"""Policy-gradient objectives on the denoising MDP.

State = (x_t, t, cond), action = x_next, one episode = one sampling trajectory,
reward only at the end. Every step of a trajectory shares its sample's advantage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from diffusion_omni.core.registry import ALGORITHMS


def compute_advantages(rewards: np.ndarray, groups: np.ndarray, mode: str = "group", eps: float = 1e-4,
                       clip: float | None = None) -> np.ndarray:
    """``group``: normalise within samples of the same prompt (GRPO). ``batch``: across the batch. ``none``: raw."""
    r = np.asarray(rewards, dtype=np.float64)
    if mode == "group":
        adv = np.zeros_like(r)
        for g in np.unique(groups):
            m = groups == g
            adv[m] = (r[m] - r[m].mean()) / (r[m].std() + eps)
    elif mode == "batch":
        adv = (r - r.mean()) / (r.std() + eps)
    elif mode == "none":
        adv = r.copy()
    else:
        raise ValueError(f"unknown advantage mode {mode!r}")
    return adv if clip is None else np.clip(adv, -clip, clip)


@dataclass
class LossOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


class PolicyGradient:
    """PPO-clipped surrogate per denoising step, plus an optional KL penalty to a reference policy."""

    def __init__(self, clip_range: float = 1e-4, kl_coef: float = 0.0, adv_mode: str = "group",
                 adv_clip: float | None = 5.0):
        self.clip_range, self.kl_coef, self.adv_mode, self.adv_clip = clip_range, kl_coef, adv_mode, adv_clip

    def advantages(self, rewards, groups) -> np.ndarray:
        return compute_advantages(rewards, groups, self.adv_mode, clip=self.adv_clip)

    def loss(self, log_prob, old_log_prob, advantages, kl=None) -> LossOutput:
        ratio = torch.exp(log_prob - old_log_prob)
        clipped = ratio.clamp(1 - self.clip_range, 1 + self.clip_range)
        pg = torch.maximum(-advantages * ratio, -advantages * clipped).mean()
        metrics = {
            "pg_loss": pg.item(),
            "ratio": ratio.mean().item(),
            "clip_frac": ((ratio - 1).abs() > self.clip_range).float().mean().item(),
        }
        loss = pg
        if kl is not None and self.kl_coef > 0:
            loss = loss + self.kl_coef * kl.mean()
            metrics["kl"] = kl.mean().item()
        return LossOutput(loss, metrics)


@ALGORITHMS.register("grpo")
class GRPO(PolicyGradient):
    """Group-relative advantages: Flow-GRPO / DanceGRPO for images and video, diffu-GRPO for text."""

    def __init__(self, clip_range: float = 1e-4, kl_coef: float = 0.0, adv_mode: str = "group", adv_clip=5.0):
        super().__init__(clip_range, kl_coef, adv_mode, adv_clip)


@ALGORITHMS.register("ddpo")
class DDPO(PolicyGradient):
    """DDPO (Black et al., 2023): batch-normalised advantages, no KL."""

    def __init__(self, clip_range: float = 1e-4, kl_coef: float = 0.0, adv_mode: str = "batch", adv_clip=5.0):
        super().__init__(clip_range, kl_coef, adv_mode, adv_clip)
