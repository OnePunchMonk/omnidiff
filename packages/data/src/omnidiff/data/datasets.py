"""SFT data. A dataset is anything with ``sample(n, rng) -> (x0, cond)``.

The built-ins are tiny synthetic sets so the whole SFT -> RL -> sample loop runs
on a laptop CPU. Point a config at your own class (``type: my_pkg.data:MyData``)
for real data.
"""

from __future__ import annotations

import numpy as np
import torch

from omnidiff.core.registry import DATASETS

PALETTE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float32)


def _blobs(centers: np.ndarray, radius: np.ndarray, H: int, W: int) -> np.ndarray:
    """Gaussian blobs, centers (..., 2) in pixels and radius broadcastable to centers[..., 0]."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dy = yy - centers[..., 0, None, None]
    dx = xx - centers[..., 1, None, None]
    r = np.asarray(radius, dtype=np.float32)[..., None, None]
    return np.exp(-(dy**2 + dx**2) / (2 * r**2)).astype(np.float32)


@DATASETS.register("blobs")
class Blobs:
    """One coloured blob on black. Class = colour."""

    def __init__(self, sample_shape, num_classes: int = 4):
        _, self.H, self.W = sample_shape
        if num_classes > len(PALETTE):
            raise ValueError(f"at most {len(PALETTE)} classes")
        self.num_classes = num_classes

    def sample(self, n, rng):
        labels = rng.integers(0, self.num_classes, n)
        centers = rng.uniform(0.25, 0.75, (n, 2)) * [self.H, self.W]
        radius = rng.uniform(0.1, 0.2, n) * min(self.H, self.W)
        img = PALETTE[labels][:, :, None, None] * _blobs(centers, radius, self.H, self.W)[:, None]
        return torch.from_numpy(img * 2 - 1), {"class_ids": torch.from_numpy(labels)}


@DATASETS.register("moving_blob")
class MovingBlob:
    """A blob moving in a straight line. Class = direction (right, left, down, up)."""

    DIRECTIONS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.float32)

    def __init__(self, sample_shape, num_classes: int = 4):
        _, self.T, self.H, self.W = sample_shape
        self.num_classes = min(num_classes, len(self.DIRECTIONS))

    def sample(self, n, rng):
        labels = rng.integers(0, self.num_classes, n)
        size = min(self.H, self.W)
        speed = rng.uniform(0.5, 1.0, n) * size / (2 * max(self.T - 1, 1))
        mid = rng.uniform(0.4, 0.6, (n, 2)) * [self.H, self.W]
        offset = np.arange(self.T) - (self.T - 1) / 2
        centers = mid[:, None] + (self.DIRECTIONS[labels] * speed[:, None])[:, None] * offset[None, :, None]
        radius = np.repeat(rng.uniform(0.1, 0.15, (n, 1)) * size, self.T, axis=1)
        colour = PALETTE[rng.integers(0, len(PALETTE), n)]
        video = colour[:, :, None, None, None] * _blobs(centers, radius, self.H, self.W)[:, None]
        return torch.from_numpy(video * 2 - 1), {"class_ids": torch.from_numpy(labels)}


TOY_LINES = [
    "the cat sat on the mat.", "a dog ran in the park.", "the sun is warm today.",
    "my friend likes green tea.", "the bird sang a song.", "a fish swims in the sea.",
    "the cat ate the fish.", "a dog found a stick.", "the sun set over the hill.",
    "my friend reads old books.", "the bird flew to the tree.", "a fish hid under a rock.",
    "the cat likes the warm sun.", "a dog and a cat play.", "the sun rose at six.",
    "my friend has a red bike.", "the bird built a nest.", "a fish jumped out of the lake.",
]


@DATASETS.register("text_lines")
class TextLines:
    """Lines of text, from a list, a file (one example per line), or the built-in toy corpus."""

    def __init__(self, codec, lines: list[str] | None = None, path: str | None = None):
        if path is not None:
            with open(path) as f:
                lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        self.lines = lines or TOY_LINES
        self.codec = codec

    def sample(self, n, rng):
        idx = rng.integers(0, len(self.lines), n)
        return self.codec.encode([self.lines[i] for i in idx]), {}
