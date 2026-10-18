"""Sample-set metrics, NumPy only.

Few-step students fail in two distinct ways: they get worse (blur, artifacts) or
they get narrower (mode collapse). One number hides which, so the scorecard
reports quality and coverage separately: Fréchet distance plus k-NN
precision / recall (Kynkäänniemi et al., 2019) in a fixed feature space, and
within-prompt diversity.

At toy scale the feature space is a seeded random ReLU projection of the pixels.
That is enough to rank samplers against each other on the same data; it is not
comparable to Inception-based FID.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


class RandomFeatures:
    def __init__(self, in_dim: int, out_dim: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.standard_normal((in_dim, out_dim)) / np.sqrt(in_dim)
        self.b = rng.uniform(-0.5, 0.5, out_dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(len(x), -1) * 2 - 1
        return np.maximum(x @ self.w + self.b, 0.0)


def _psd_sqrt(m: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((m + m.T) / 2)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.T


def frechet_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    r = _psd_sqrt(s1)
    cross = np.sqrt(np.clip(np.linalg.eigvalsh(r @ s2 @ r), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * cross)


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a**2).sum(1)[:, None] + (b**2).sum(1)[None] - 2 * a @ b.T
    return np.sqrt(np.clip(d, 0, None))


def precision_recall(real: np.ndarray, fake: np.ndarray, k: int = 3) -> tuple[float, float]:
    """Precision: share of fake samples inside the real k-NN manifold. Recall: the reverse."""

    def radii(x):
        d = _pairwise(x, x)
        return np.sort(d, axis=1)[:, k]  # column 0 is the point itself

    r_real, r_fake = radii(real), radii(fake)
    d = _pairwise(fake, real)
    precision = (d <= r_real[None]).any(1).mean()
    recall = (d.T <= r_fake[None]).any(1).mean()
    return float(precision), float(recall)


def within_prompt_diversity(samples, prompts) -> float:
    """Mean pairwise distance between samples that share a prompt (RMS per element for arrays,
    share of distinct strings for text)."""
    groups = defaultdict(list)
    for i, p in enumerate(prompts):
        groups[repr(p)].append(i)
    scores = []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        if isinstance(samples, list):
            scores.append(len({samples[i] for i in idx}) / len(idx))
        else:
            x = np.asarray(samples)[idx].reshape(len(idx), -1).astype(np.float64)
            d = _pairwise(x, x) / np.sqrt(x.shape[1])
            scores.append(d[np.triu_indices(len(idx), 1)].mean())
    return float(np.mean(scores)) if scores else float("nan")


def set_metrics(samples, prompts, reference=None, features: RandomFeatures | None = None, prefix: str = "") -> dict:
    """Metrics for one sample set; ``reference`` (same modality) adds FD and precision/recall against it."""
    out = {"diversity": within_prompt_diversity(samples, prompts)}
    if reference is not None and not isinstance(samples, list):
        f = features or RandomFeatures(int(np.prod(np.asarray(samples).shape[1:])))
        fs, fr = f(samples), f(reference)
        out[f"{prefix}fd"] = frechet_distance(fs, fr)
        out[f"{prefix}precision"], out[f"{prefix}recall"] = precision_recall(fr, fs)
    return out
