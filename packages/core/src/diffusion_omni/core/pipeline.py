"""Inference entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .build import Components, build
from .config import apply_overrides, load_config
from .sampling import Trajectory, sample
from .utils import CONFIG, load_weights, make_generator


@dataclass
class Generation:
    samples: object            # decoded: numpy array for pixels, list[str] for text
    prompts: list
    raw: torch.Tensor           # the process-space tensor
    trajectory: Trajectory | None = None


class Pipeline:
    def __init__(self, components: Components):
        self.c = components
        self.c.model.eval()

    @classmethod
    def from_config(cls, cfg: dict, device=None) -> "Pipeline":
        return cls(build(cfg, device))

    @classmethod
    def from_checkpoint(cls, path, device=None, overrides=()) -> "Pipeline":
        path = Path(path)
        cfg = apply_overrides(load_config(path / CONFIG), overrides)
        pipe = cls.from_config(cfg, device)
        pipe.c.model.load_state_dict(load_weights(path, pipe.c.device))
        return pipe

    @torch.no_grad()
    def generate(self, prompts: list | None = None, n: int | None = None, num_steps: int | None = None,
                 guidance_scale: float | None = None, seed: int = 0, return_trajectory: bool = False,
                 timesteps=None) -> Generation:
        """``num_steps`` alone uses the process's default grid; otherwise ``timesteps`` or the config's grid."""
        c = self.c
        if timesteps is None and num_steps is None:
            timesteps = c.sampling.timesteps
        if prompts is None:
            prompts = c.prompts.sample(n or 1, np.random.default_rng(seed))
        cond = c.prompts.encode(prompts, c.device)
        out = sample(
            c.model, c.process, (len(prompts), *c.sample_shape), cond,
            num_steps or c.sampling.num_steps,
            c.sampling.guidance_scale if guidance_scale is None else guidance_scale,
            generator=make_generator(seed, c.device), keep_trajectory=return_trajectory, timesteps=timesteps,
        )
        traj = out if return_trajectory else None
        x = traj.samples if return_trajectory else out
        return Generation(c.codec.decode(x), list(prompts), x, traj)
