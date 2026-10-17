from __future__ import annotations

import torch


class Distiller:
    """A distillation method: which roles it trains, in what order, and with which losses.

    ``loss(role, tr)`` draws its own data from the trainer (``tr.data.batch``,
    ``tr.prompt_cond``, ``tr.student_rollout``) and returns ``(loss, metrics)`` for
    that role. The trainer handles freezing, optimiser steps and EMA.
    """

    processes: tuple[str, ...] = ("gaussian", "flow", "masked")
    sampler: str = "renoise"            # default student sampler for continuous processes
    discrete_sampler: str = "ancestral"  # ... and for masked diffusion
    teacher_sampler: str | None = None  # teacher pool sampler; default: ode / ancestral
    two_time: bool = False              # whether the student takes the target time s (flow maps)
    teacher_process_defaults: dict = {}  # process settings for the teacher, applied before the pool is drawn
    student_process_defaults: dict = {}  # process settings for the student, exported with it
    new_param_prefixes: tuple[str, ...] = ("s_embed.",)  # student parameters the teacher does not have

    def student_spec(self, spec: dict, tr) -> dict:
        """The student's model spec, given the teacher's (plus ``two_time``)."""
        return spec

    def setup(self, tr) -> None:
        pass

    def state_dict(self) -> dict:
        """Return method-owned state that is not part of a trainer role."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore method-owned state that is not part of a trainer role."""
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
