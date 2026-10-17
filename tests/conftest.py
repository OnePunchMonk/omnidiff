from pathlib import Path

import pytest

from omnidiff.config import load_config

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))

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


def tiny_config(path, tmp_path, *extra):
    return load_config(path, [*TINY, f"output_dir={tmp_path}", *extra])


@pytest.fixture(params=CONFIGS, ids=lambda p: p.stem)
def config_path(request):
    return request.param
