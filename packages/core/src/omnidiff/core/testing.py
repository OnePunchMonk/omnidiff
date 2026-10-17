"""Helpers for testing configs and plugins at toy size on CPU."""

from __future__ import annotations

from .config import load_config

TINY = [
    "device=cpu",
    "model.dim=32",
    "model.backbone.depth=1",
    "model.backbone.heads=2",
    "sampling.num_steps=4",
    "sft.steps=3",
    "sft.batch_size=4",
    "sft.log_every=1",
    "rl.init_from=null",
    "rl.epochs=1",
    "rl.prompts_per_epoch=2",
    "rl.group_size=2",
    "rl.rollout_batch_size=3",
    "rl.train_batch_size=2",
]


def tiny_config(path, output_dir, *extra) -> dict:
    """Load a config shrunk to a few seconds of CPU work, writing under ``output_dir``."""
    return load_config(path, [*TINY, f"output_dir={output_dir}", *extra])
