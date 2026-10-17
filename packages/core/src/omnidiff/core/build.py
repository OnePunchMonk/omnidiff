"""Config -> components. The one place that knows how the pieces are wired."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import SamplingConfig, from_dict
from .nn.denoiser import build_model
from .registry import CODECS, PROCESSES, PROMPTS, load_plugins
from .utils import resolve_device


@dataclass
class Components:
    cfg: dict
    device: torch.device
    sample_shape: tuple[int, ...]
    codec: object
    process: object
    model: torch.nn.Module
    prompts: object
    sampling: SamplingConfig
    context: dict  # shared kwargs offered to every registry build


def build(cfg: dict, device=None) -> Components:
    load_plugins(cfg.get("plugins", []))
    device = resolve_device(device or cfg.get("device", "auto"))
    shape = tuple(cfg["sample_shape"])
    codec = CODECS.build(cfg.get("codec", "pixel"), sample_shape=shape)
    context = {"sample_shape": shape, "codec": codec}
    for attr in ("mask_id", "vocab_size"):
        if hasattr(codec, attr):
            context[attr] = getattr(codec, attr)
    process = PROCESSES.build(cfg["process"], **context)
    model = build_model(cfg["model"], **context).to(device)
    prompts = PROMPTS.build(cfg.get("prompts", "none"), **context)
    sampling = from_dict(SamplingConfig, cfg.get("sampling"))
    return Components(cfg, device, shape, codec, process, model, prompts, sampling, context)
