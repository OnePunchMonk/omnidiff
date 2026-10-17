"""Modality adapters: native tensor -> tokens (B, N, D) -> model output.

The backbone never sees a modality. Swapping text for image for video is a matter
of picking an adapter; everything downstream of it is shared.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import ADAPTERS
from .layers import sincos_nd


class Adapter(nn.Module):
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def unembed(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Map backbone tokens back to the output for input ``x`` (same shape for continuous data, logits for tokens)."""
        raise NotImplementedError


@ADAPTERS.register("tokens")
class TokenAdapter(Adapter):
    """Token ids (B, L) -> logits (B, L, vocab_size). Used with masked (discrete) diffusion."""

    def __init__(self, dim: int, vocab_size: int, sample_shape: tuple[int, ...]):
        super().__init__()
        (seq_len,) = sample_shape
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)
        self.head = nn.Linear(dim, vocab_size)
        nn.init.normal_(self.tok.weight, std=0.02)

    def embed(self, x):
        return self.tok(x) + self.pos[:, : x.shape[1]]

    def unembed(self, h, x):
        return self.head(h)


@ADAPTERS.register("image_patch")
class ImagePatchAdapter(Adapter):
    """Images (B, C, H, W) as non-overlapping p x p patches."""

    def __init__(self, dim: int, sample_shape: tuple[int, ...], patch: int = 2):
        super().__init__()
        C, H, W = sample_shape
        if H % patch or W % patch:
            raise ValueError(f"image {H}x{W} not divisible by patch {patch}")
        self.p = patch
        self.proj = nn.Linear(C * patch * patch, dim)
        self.head = nn.Linear(dim, C * patch * patch)
        self.register_buffer("pos", sincos_nd(dim, (H // patch, W // patch))[None], persistent=False)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def embed(self, x):
        B, C, H, W = x.shape
        p = self.p
        x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)
        return self.proj(x) + self.pos

    def unembed(self, h, x):
        B, C, H, W = x.shape
        p = self.p
        out = self.head(h).reshape(B, H // p, W // p, C, p, p)
        return out.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


@ADAPTERS.register("video_patch")
class VideoPatchAdapter(Adapter):
    """Videos (B, C, T, H, W) as pt x p x p spatio-temporal patches, attended jointly."""

    def __init__(self, dim: int, sample_shape: tuple[int, ...], patch: int = 2, temporal_patch: int = 1):
        super().__init__()
        C, T, H, W = sample_shape
        if T % temporal_patch or H % patch or W % patch:
            raise ValueError(f"video {T}x{H}x{W} not divisible by patch {temporal_patch}x{patch}x{patch}")
        self.p, self.pt = patch, temporal_patch
        n = C * temporal_patch * patch * patch
        self.proj = nn.Linear(n, dim)
        self.head = nn.Linear(dim, n)
        grid = (T // temporal_patch, H // patch, W // patch)
        self.register_buffer("pos", sincos_nd(dim, grid)[None], persistent=False)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def embed(self, x):
        B, C, T, H, W = x.shape
        p, pt = self.p, self.pt
        x = x.reshape(B, C, T // pt, pt, H // p, p, W // p, p).permute(0, 2, 4, 6, 1, 3, 5, 7)
        return self.proj(x.reshape(B, -1, C * pt * p * p)) + self.pos

    def unembed(self, h, x):
        B, C, T, H, W = x.shape
        p, pt = self.p, self.pt
        out = self.head(h).reshape(B, T // pt, H // p, W // p, C, pt, p, p)
        return out.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B, C, T, H, W)
