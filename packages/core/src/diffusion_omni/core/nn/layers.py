"""Transformer building blocks shared by every modality."""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedding(nn.Module):
    """Sinusoidal features of t in [0, 1], followed by an MLP."""

    def __init__(self, dim: int, freq_dim: int = 256, max_period: float = 10000.0, scale: float = 1000.0):
        super().__init__()
        self.freq_dim, self.max_period, self.scale = freq_dim, max_period, scale
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = (t.float() * self.scale)[:, None] * freqs[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


def sincos_1d(dim: int, positions: torch.Tensor) -> torch.Tensor:
    omega = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float64) / (dim / 2))
    out = positions.double()[:, None] * omega[None]
    return torch.cat([out.sin(), out.cos()], dim=1).float()


def sincos_nd(dim: int, grid: tuple[int, ...]) -> torch.Tensor:
    """Fixed sin-cos embedding for an N-d token grid, (prod(grid), dim). Axes share dim evenly."""
    per_axis = (dim // len(grid)) // 2 * 2
    coords = torch.meshgrid(*[torch.arange(s) for s in grid], indexing="ij")
    emb = torch.cat([sincos_1d(per_axis, c.reshape(-1)) for c in coords], dim=1)
    return F.pad(emb, (0, dim - emb.shape[1]))


_MATH_ATTENTION = False


@contextlib.contextmanager
def math_attention():
    """Use explicit softmax attention. Forward-mode AD (``torch.func.jvp``) works through it; the fused SDPA kernels have no JVP rule."""
    global _MATH_ATTENTION
    prev, _MATH_ATTENTION = _MATH_ATTENTION, True
    try:
        yield
    finally:
        _MATH_ATTENTION = prev


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        if _MATH_ATTENTION:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            if attn_mask is not None:
                scores = scores + attn_mask if attn_mask.dtype != torch.bool else scores.masked_fill(~attn_mask, float("-inf"))
            out = scores.softmax(-1) @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class DiTBlock(nn.Module):
    """Bidirectional transformer block with adaLN-Zero conditioning (Peebles & Xie, 2023)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c).chunk(6, dim=-1)
        x = x + gate1.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift1, scale1))
        x = x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x
