from __future__ import annotations

from omnidiff.core.cli import add_config_command
from omnidiff.core.config import load_config


def _distill(args):
    from .trainer import DistillTrainer

    path = DistillTrainer(load_config(args.config, args.overrides)).train()
    print(f"saved {path}")


def register(sub) -> None:
    add_config_command(sub, "distill", "few-step distillation of a teacher checkpoint", _distill)
