from __future__ import annotations

from diffusion_omni.core.cli import add_config_command
from diffusion_omni.core.config import load_config


def _run(trainer_cls):
    def fn(args):
        path = trainer_cls(load_config(args.config, args.overrides)).train()
        print(f"saved {path}")

    return fn


def register(sub) -> None:
    from .rl.trainer import RLTrainer
    from .sft import SFTTrainer

    add_config_command(sub, "sft", "supervised denoising training from a config", _run(SFTTrainer))
    add_config_command(sub, "rl", "online RL post-training from a config", _run(RLTrainer))
