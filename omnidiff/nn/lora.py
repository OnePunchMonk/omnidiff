"""Minimal LoRA. With adapters disabled the model is the frozen base, which doubles as the KL reference."""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("qkv", "proj", "fc1", "fc2")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.lora_a.to(base.weight)
        self.lora_b.to(base.weight)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + self.lora_b(self.lora_a(self.dropout(x))) * self.scale
        return y

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + self.scale * self.lora_b.weight @ self.lora_a.weight


def apply_lora(model: nn.Module, rank: int = 8, alpha: float | None = None, dropout: float = 0.0,
               targets=DEFAULT_TARGETS) -> int:
    """Freeze ``model`` and wrap every ``nn.Linear`` whose attribute name is in ``targets``."""
    model.requires_grad_(False)
    replaced = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, rank, alpha or rank, dropout))
                replaced += 1
    if not replaced:
        raise ValueError(f"no nn.Linear named any of {targets} found")
    return replaced


def has_lora(model: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in model.modules())


@contextlib.contextmanager
def lora_disabled(model: nn.Module):
    layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    prev = [m.enabled for m in layers]
    for m in layers:
        m.enabled = False
    try:
        yield model
    finally:
        for m, e in zip(layers, prev):
            m.enabled = e


def merged_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """State dict with LoRA folded in, loadable into the same architecture without adapters."""
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for name, m in model.named_modules():
        if not isinstance(m, LoRALinear):
            continue
        prefix = f"{name}." if name else ""
        for k in ("base.weight", "base.bias", "lora_a.weight", "lora_b.weight"):
            state.pop(prefix + k, None)
        state[prefix + "weight"] = m.merged_weight().detach().clone()
        if m.base.bias is not None:
            state[prefix + "bias"] = m.base.bias.detach().clone()
    return state
