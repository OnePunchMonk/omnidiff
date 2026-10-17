"""The model contract every process relies on.

Any ``nn.Module`` with ``forward(x, t, cond, drop_cond) -> Tensor`` and a
``supports_guidance`` attribute works; register it under ``MODELS`` to use it from
a config. ``Denoiser`` is the default: adapter -> DiT backbone -> adapter.

Two-time models (``two_time = True``) also take ``s``, the time the step lands on.
Flow-map students use it; the extra embedding is zero-initialised, so a two-time
model built from one-time weights starts out ignoring ``s``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import ADAPTERS, BACKBONES, CONDITIONERS, MODELS
from .layers import TimestepEmbedding


class Denoiser(nn.Module):
    def __init__(self, adapter: nn.Module, backbone: nn.Module, conditioner: nn.Module, dim: int,
                 two_time: bool = False):
        super().__init__()
        self.adapter, self.backbone, self.conditioner = adapter, backbone, conditioner
        self.t_embed = TimestepEmbedding(dim)
        self.two_time = two_time
        if two_time:
            self.s_embed = TimestepEmbedding(dim)  # embeds the gap t - s
            nn.init.zeros_(self.s_embed.mlp[-1].weight)
            nn.init.zeros_(self.s_embed.mlp[-1].bias)

    @property
    def supports_guidance(self) -> bool:
        return getattr(self.conditioner, "supports_guidance", False)

    def features(self, x, t, cond=None, drop_cond=None, s=None) -> torch.Tensor:
        """Backbone output tokens (B, N, D), before the output head. Discriminator heads read these."""
        B = x.shape[0]
        t = _per_sample(t, B, x.device)
        c = self.t_embed(t) + self.conditioner(cond or {}, B, x.device, drop_cond)
        if self.two_time:
            c = c + self.s_embed(t - (t if s is None else _per_sample(s, B, x.device)))
        noise_key = getattr(self.adapter, "noise_key", None)
        tokens = self.adapter.embed(x, (cond or {}).get(noise_key)) if noise_key else self.adapter.embed(x)
        return self.backbone(tokens, c)

    def forward(self, x, t, cond=None, drop_cond=None, s=None):
        return self.adapter.unembed(self.features(x, t, cond, drop_cond, s), x)


def _per_sample(t, batch_size: int, device) -> torch.Tensor:
    t = torch.as_tensor(t, device=device, dtype=torch.float32)
    return t.expand(batch_size) if t.dim() == 0 else t


@MODELS.register("denoiser")
def denoiser(dim: int, adapter, backbone="dit", conditioner="none", two_time: bool = False, **context) -> Denoiser:
    context["dim"] = dim
    return Denoiser(
        ADAPTERS.build(adapter, **context),
        BACKBONES.build(backbone, **context),
        CONDITIONERS.build(conditioner, **context),
        dim,
        two_time,
    )


def build_model(spec: dict, **context) -> nn.Module:
    spec = dict(spec)
    spec.setdefault("type", "denoiser")
    return MODELS.build(spec, **context)
