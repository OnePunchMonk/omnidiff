"""Rewards: ``reward(samples, prompts) -> np.ndarray`` of shape (B,).

``samples`` is decoded data (numpy in [0, 1] for pixels, a list of str for
text). The built-ins are cheap toy objectives; real reward models (aesthetic
predictors, VLM judges, verifiers) are registered the same way.

A reward that is differentiable also defines ``torch_call(x, prompts=None)``: the
same function on a torch tensor in [0, 1], returning a (B,) tensor that keeps the
autograd graph, and sets ``differentiable = True``. Methods that need reward gradients (``dmd2_reward``) use it.
"""

from __future__ import annotations

import numpy as np
import torch

from omnidiff.core.registry import REWARDS


def _per_sample_mean(x):
    return x.reshape(x.shape[0], -1).mean(1)


@REWARDS.register("channel_dominance")
class ChannelDominance:
    """Mean of one colour channel minus the mean of the others ("make it red" for channel 0)."""

    differentiable = True

    def __init__(self, channel: int = 0):
        self.channel = channel

    def __call__(self, samples, prompts):
        x = np.asarray(samples)
        others = np.delete(x, self.channel, axis=1)
        return _per_sample_mean(x[:, self.channel]) - _per_sample_mean(others)

    def torch_call(self, x: torch.Tensor, prompts=None) -> torch.Tensor:
        keep = [c for c in range(x.shape[1]) if c != self.channel]
        return _per_sample_mean(x[:, self.channel]) - _per_sample_mean(x[:, keep])


@REWARDS.register("brightness")
class Brightness:
    """Mean intensity, or closeness to ``target`` if given."""

    differentiable = True

    def __init__(self, target: float | None = None):
        self.target = target

    def __call__(self, samples, prompts):
        m = _per_sample_mean(np.asarray(samples))
        return m if self.target is None else -np.abs(m - self.target)

    def torch_call(self, x: torch.Tensor, prompts=None) -> torch.Tensor:
        m = _per_sample_mean(x)
        return m if self.target is None else -(m - self.target).abs()


@REWARDS.register("temporal_smoothness")
class TemporalSmoothness:
    """Negative mean absolute frame difference for videos (B, C, T, H, W)."""

    differentiable = True

    def __call__(self, samples, prompts):
        return -_per_sample_mean(np.abs(np.diff(np.asarray(samples), axis=2)))

    def torch_call(self, x: torch.Tensor, prompts=None) -> torch.Tensor:
        return -_per_sample_mean(torch.diff(x, dim=2).abs())


@REWARDS.register("char_fraction")
class CharFraction:
    """Fraction of generated characters that are in ``chars``."""

    def __init__(self, chars: str = "aeiou", completion_only: bool = True):
        self.chars, self.completion_only = set(chars), completion_only

    def __call__(self, samples, prompts):
        out = []
        for text, prompt in zip(samples, prompts):
            if self.completion_only and prompt and text.startswith(prompt):
                text = text[len(prompt):]
            out.append(sum(c in self.chars for c in text) / len(text) if text else 0.0)
        return np.asarray(out)


@REWARDS.register("contains")
class Contains:
    def __init__(self, substring: str):
        self.substring = substring

    def __call__(self, samples, prompts):
        return np.asarray([float(self.substring in s) for s in samples])


@REWARDS.register("composite")
class Composite:
    """Weighted sum: ``terms: [{weight: 1.0, reward: {type: ...}}, ...]``."""

    def __init__(self, terms: list[dict], **context):
        self.terms = [(float(t.get("weight", 1.0)), REWARDS.build(t["reward"], **context)) for t in terms]

    def __call__(self, samples, prompts):
        return sum(w * np.asarray(r(samples, prompts), dtype=np.float64) for w, r in self.terms)

    @property
    def differentiable(self) -> bool:
        return all(getattr(r, "differentiable", False) for _, r in self.terms)

    def torch_call(self, x: torch.Tensor, prompts=None) -> torch.Tensor:
        return sum(w * r.torch_call(x, prompts) for w, r in self.terms)
