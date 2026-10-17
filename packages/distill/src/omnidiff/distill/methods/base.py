from __future__ import annotations

import torch


class Distiller:
    """A distillation method: which roles it trains, in what order, and with which losses.

    ``loss(role, tr)`` draws its own data from the trainer (``tr.data.batch``,
    ``tr.prompt_cond``, ``tr.student_rollout``) and returns ``(loss, metrics)`` for
    that role. The trainer handles freezing, optimiser steps and EMA.
    """

    processes: tuple[str, ...] = ("gaussian", "flow", "masked")
    sampler: str = "renoise"   # default student sampler for continuous processes
    two_time: bool = False     # whether the student takes the target time s (flow maps)

    def setup(self, tr) -> None:
        pass

    def default_grid(self, tr) -> torch.Tensor:
        return tr.process.timesteps(4, tr.device)

    def schedule(self, it: int, tr) -> list[str]:
        return ["student"]

    def loss(self, role: str, tr) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def after_update(self, role: str, tr) -> None:
        pass


def per_sample_mean(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).mean(1)
