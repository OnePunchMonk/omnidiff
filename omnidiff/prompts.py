"""Prompt sources: ``sample(n, rng) -> list`` of prompts and ``encode(prompts, device) -> cond``.

A prompt is whatever a reward needs to see next to the sample (a class id, a
string, None). ``encode`` turns a list of them into the model's ``cond`` dict.
"""

from __future__ import annotations

import torch

from .registry import PROMPTS


@PROMPTS.register("none")
class Unconditional:
    def sample(self, n, rng):
        return [None] * n

    def encode(self, prompts, device=None):
        return {}


@PROMPTS.register("class")
class ClassPrompts:
    def __init__(self, num_classes: int, classes: list[int] | None = None):
        self.classes = list(classes) if classes is not None else list(range(num_classes))

    def sample(self, n, rng):
        return [int(c) for c in rng.choice(self.classes, n)]

    def encode(self, prompts, device=None):
        return {"class_ids": torch.tensor(prompts, dtype=torch.long, device=device)}


@PROMPTS.register("text")
class TextPrompts:
    """Text prefixes for masked diffusion. Encoded as fixed leading positions in ``cond['prefix']``."""

    def __init__(self, codec, sample_shape, prompts: list[str] | None = None, path: str | None = None):
        if path is not None:
            with open(path) as f:
                prompts = [ln.rstrip("\n") for ln in f if ln.strip()]
        if not prompts:
            raise ValueError("text prompts need `prompts` or `path`")
        self.prompts, self.codec = list(prompts), codec
        (self.seq_len,) = sample_shape

    def sample(self, n, rng):
        return [str(p) for p in rng.choice(self.prompts, n)]

    def encode(self, prompts, device=None):
        prefix = torch.full((len(prompts), self.seq_len), -1, dtype=torch.long)
        for i, p in enumerate(prompts):
            ids = self.codec.tokenizer.encode(p)[: self.seq_len]
            prefix[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return {"prefix": prefix.to(device)}
