"""SFT and distillation data. A dataset is anything with ``sample(n, rng) -> (x0, cond)``.

The built-ins are tiny synthetic sets so the whole SFT -> RL -> distill loop runs
on a laptop CPU. Because they are synthetic, each one also knows exactly what a
correct sample looks like: ``oracle(samples, prompts) -> dict`` scores generated
samples against the true generating process (class/direction accuracy, shape
error, mode coverage). The scorecard reports these next to the generic metrics.
Point a config at your own class (``type: my_pkg.data:MyData``) for real data.
"""

from __future__ import annotations

import numpy as np
import torch
from diffusion_omni.core.registry import DATASETS

PALETTE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float32)


def _fit_blob(g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intensity-weighted centroid (..., 2), radius (...) and the ideal blob they imply, for maps (..., H, W)."""
    H, W = g.shape[-2:]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    m = np.clip(g, 0, None)
    mass = m.sum((-2, -1)) + 1e-8
    cy = (m * yy).sum((-2, -1)) / mass
    cx = (m * xx).sum((-2, -1)) / mass
    var = (m * ((yy - cy[..., None, None]) ** 2 + (xx - cx[..., None, None]) ** 2)).sum((-2, -1)) / mass / 2
    radius = np.sqrt(np.clip(var, 0.25, None))
    centers = np.stack([cy, cx], -1)
    return centers, radius, _blobs(centers, radius, H, W)


def _center_histogram(centers: np.ndarray, H: int, W: int, bins: int = 4) -> np.ndarray:
    h, _, _ = np.histogram2d(centers[:, 0] / H, centers[:, 1] / W, bins=bins, range=[[0, 1], [0, 1]])
    return h.ravel() / max(h.sum(), 1)


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

    def _analyse(self, x: np.ndarray, classes: np.ndarray):
        colours = PALETTE[: self.num_classes]
        proj = np.einsum("nchw,kc->nkhw", x, colours) / (colours**2).sum(1)[None, :, None, None]
        # classify by the palette colour whose rank-1 fit leaves the least residual (yellow = red + green,
        # so projection energy alone cannot tell them apart)
        fit = proj[:, :, None] * colours[None, :, :, None, None]
        pred = ((x[:, None] - fit) ** 2).sum((2, 3, 4)).argmin(1)
        g = proj[np.arange(len(x)), classes]
        centers, _, ideal = _fit_blob(g)
        residual = x - colours[classes][:, :, None, None] * g[:, None]
        return pred, g, centers, ideal, residual

    def oracle(self, samples, prompts) -> dict:
        """class_acc: dominant palette colour is the prompted one. colour_err: off-colour energy.
        shape_err: distance from the best-fitting single blob. center_tv: total variation between
        the histogram of blob centres and the data's (mode coverage)."""
        x = np.asarray(samples, dtype=np.float64)
        classes = np.asarray(prompts, dtype=int)
        pred, g, centers, ideal, residual = self._analyse(x, classes)
        real, cond = self.sample(max(len(x), 512), np.random.default_rng(1234))
        real_centers = self._analyse(((real.numpy() + 1) / 2).astype(np.float64), cond["class_ids"].numpy())[2]
        tv = 0.5 * np.abs(_center_histogram(centers, self.H, self.W) - _center_histogram(real_centers, self.H, self.W)).sum()
        return {
            "class_acc": float((pred == classes).mean()),
            "colour_err": float(np.abs(residual).mean()),
            "shape_err": float(np.abs(g - ideal).mean()),
            "center_tv": float(tv),
        }


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

    def oracle(self, samples, prompts) -> dict:
        """direction_acc: net motion points the prompted way. shape_err: per-frame distance from the best
        single blob. motion_err: deviation of the centre track from a straight constant-speed line (pixels)."""
        x = np.asarray(samples, dtype=np.float64)
        g = x.max(1)  # colour-agnostic intensity, (n, T, H, W)
        centers, _, ideal = _fit_blob(g)
        disp = centers[:, -1] - centers[:, 0]
        pred = (disp @ self.DIRECTIONS[: self.num_classes].T).argmax(1)
        offset = np.arange(self.T) - (self.T - 1) / 2
        slope = (centers * offset[None, :, None]).sum(1, keepdims=True) / (offset**2).sum()
        line = centers.mean(1, keepdims=True) + slope * offset[None, :, None]
        return {
            "direction_acc": float((pred == np.asarray(prompts, dtype=int)).mean()),
            "shape_err": float(np.abs(g - ideal).mean()),
            "motion_err": float(np.linalg.norm(centers - line, axis=-1).mean()),
        }


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

    def oracle(self, samples, prompts) -> dict:
        """valid: the sample is exactly a corpus line. coverage: share of corpus lines consistent with
        each prompt that were generated at least once (mode coverage), averaged over prompts."""
        lines = set(self.lines)
        valid = [s in lines for s in samples]
        by_prompt: dict[str, set] = {}
        for s, p, ok in zip(samples, prompts, valid):
            by_prompt.setdefault(p or "", set())
            if ok:
                by_prompt[p or ""].add(s)
        coverage = []
        for p, hit in by_prompt.items():
            targets = {ln for ln in lines if ln.startswith(p)}
            if targets:
                coverage.append(len(hit & targets) / len(targets))
        return {"valid": float(np.mean(valid)), "coverage": float(np.mean(coverage)) if coverage else float("nan")}
