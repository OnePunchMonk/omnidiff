from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import BACKBONES
from .layers import DiTBlock, modulate


@BACKBONES.register("dit")
class DiT(nn.Module):
    """adaLN-Zero transformer over (B, N, D) tokens, conditioned on a (B, D) vector."""

    def __init__(self, dim: int, depth: int = 4, heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.blocks = nn.ModuleList(DiTBlock(dim, heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # adaLN-Zero: every block starts as the identity.
        for m in [b.ada[-1] for b in self.blocks] + [self.ada[-1]]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, c)
        shift, scale = self.ada(c).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)
