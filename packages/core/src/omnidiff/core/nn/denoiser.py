"""The model contract every process relies on.

Any ``nn.Module`` with ``forward(x, t, cond, drop_cond) -> Tensor`` and a
``supports_guidance`` attribute works; register it under ``MODELS`` to use it from
a config. ``Denoiser`` is the default: adapter -> DiT backbone -> adapter.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import ADAPTERS, BACKBONES, CONDITIONERS, MODELS
from .layers import TimestepEmbedding


class Denoiser(nn.Module):
    def __init__(self, adapter: nn.Module, backbone: nn.Module, conditioner: nn.Module, dim: int):
        super().__init__()
        self.adapter, self.backbone, self.conditioner = adapter, backbone, conditioner
        self.t_embed = TimestepEmbedding(dim)

    @property
    def supports_guidance(self) -> bool:
        return getattr(self.conditioner, "supports_guidance", False)

    def forward(self, x, t, cond=None, drop_cond=None):
        B = x.shape[0]
        t = torch.as_tensor(t, device=x.device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.expand(B)
        c = self.t_embed(t) + self.conditioner(cond or {}, B, x.device, drop_cond)
        return self.adapter.unembed(self.backbone(self.adapter.embed(x), c), x)


@MODELS.register("denoiser")
def denoiser(dim: int, adapter, backbone="dit", conditioner="none", **context) -> Denoiser:
    context["dim"] = dim
    return Denoiser(
        ADAPTERS.build(adapter, **context),
        BACKBONES.build(backbone, **context),
        CONDITIONERS.build(conditioner, **context),
        dim,
    )


def build_model(spec: dict, **context) -> nn.Module:
    spec = dict(spec)
    spec.setdefault("type", "denoiser")
    return MODELS.build(spec, **context)
