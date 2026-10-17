"""Registering your own components. Load it from any config with

    plugins: [examples.custom_plugin]

and then refer to the names below, e.g. ``rl.reward: {type: centered_blob}``.
Anything importable also works without registering: ``type: my_pkg.mod:MyClass``.
"""

import numpy as np
import torch.nn as nn

from omnidiff.core.registry import CONDITIONERS, REWARDS


@REWARDS.register("centered_blob")
class CenteredBlob:
    """Reward mass near the image centre. Rewards get decoded numpy samples and the prompts."""

    def __init__(self, sample_shape):  # context kwargs are injected when the signature asks for them
        H, W = sample_shape[-2:]
        yy, xx = np.mgrid[0:H, 0:W]
        self.weight = np.exp(-((yy - H / 2) ** 2 + (xx - W / 2) ** 2) / (2 * (min(H, W) / 6) ** 2))

    def __call__(self, samples, prompts):
        intensity = np.asarray(samples).mean(axis=1)  # (B, [T,] H, W)
        if intensity.ndim == 4:
            intensity = intensity.mean(axis=1)
        return (intensity * self.weight).sum((1, 2)) / intensity.sum((1, 2)).clip(min=1e-6)


@CONDITIONERS.register("class_plus_vector")
class ClassPlusVector(nn.Module):
    """A conditioner is any module mapping (cond, batch_size, device, drop) -> (B, dim)."""

    supports_guidance = True

    def __init__(self, dim, num_classes, in_dim):
        super().__init__()
        self.cls = nn.Embedding(num_classes + 1, dim)
        self.vec = nn.Linear(in_dim, dim)
        self.num_classes = num_classes

    def forward(self, cond, batch_size, device, drop=None):
        ids = cond["class_ids"]
        vec = self.vec(cond["embedding"])
        if drop is not None:
            ids = ids.masked_fill(drop, self.num_classes)
            vec = vec.masked_fill(drop[:, None], 0.0)
        return self.cls(ids) + vec
