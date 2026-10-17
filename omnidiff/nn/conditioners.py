"""Conditioners turn ``cond`` (a dict of tensors) into a (B, D) vector added to the timestep embedding.

``drop`` (bool, (B,)) swaps in the null condition per sample. It is used for
classifier-free guidance at sampling time and for label dropout during SFT.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import CONDITIONERS


@CONDITIONERS.register("none")
class NullConditioner(nn.Module):
    supports_guidance = False

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, cond, batch_size, device, drop=None):
        return torch.zeros(batch_size, self.dim, device=device)


@CONDITIONERS.register("class")
class ClassConditioner(nn.Module):
    """Class-label embedding. Index ``num_classes`` is the null label."""

    supports_guidance = True

    def __init__(self, dim: int, num_classes: int, key: str = "class_ids"):
        super().__init__()
        self.num_classes, self.key = num_classes, key
        self.emb = nn.Embedding(num_classes + 1, dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, cond, batch_size, device, drop=None):
        ids = cond[self.key]
        if drop is not None:
            ids = torch.where(drop, torch.full_like(ids, self.num_classes), ids)
        return self.emb(ids)


@CONDITIONERS.register("vector")
class VectorConditioner(nn.Module):
    """A precomputed embedding (e.g. from any external text encoder) projected to the model width."""

    supports_guidance = True

    def __init__(self, dim: int, in_dim: int, key: str = "embedding"):
        super().__init__()
        self.key = key
        self.proj = nn.Linear(in_dim, dim)
        self.null = nn.Parameter(torch.zeros(dim))

    def forward(self, cond, batch_size, device, drop=None):
        c = self.proj(cond[self.key])
        if drop is not None:
            c = torch.where(drop[:, None], self.null.expand_as(c), c)
        return c
