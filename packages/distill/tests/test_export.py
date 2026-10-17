import json
from pathlib import Path

import numpy as np
import pytest
import torch

from omnidiff.core import Pipeline
from omnidiff.core.config import load_config
from omnidiff.core.testing import tiny_config
from omnidiff.distill import DistillTrainer
from omnidiff.distill.export import SAMPLER_REQUIRED, export_bundle, load_bundle
from omnidiff.posttrain import SFTTrainer

ROOT = Path(__file__).resolve().parents[3]
TINY_DISTILL = ["distill.steps=2", "distill.batch_size=4", "distill.pool_size=8", "distill.teacher_steps=4",
                "distill.log_every=1", "distill.eval_every=2", "distill.eval_n=4"]


@pytest.fixture(scope="module")
def teachers(tmp_path_factory):
    cache = {}

    def get(stem):
        if stem not in cache:
            out = tmp_path_factory.mktemp(stem)
            cache[stem] = SFTTrainer(tiny_config(ROOT / "configs" / f"{stem}.yaml", out)).train()
        return cache[stem]

    return get


def distill_config(path, teachers, tmp_path, *extra):
    cfg = load_config(path, ["device=cpu", f"output_dir={tmp_path}", *TINY_DISTILL, *extra])
    cfg["teacher"] = str(teachers(Path(cfg["teacher"]).parts[1]))
    return cfg


CASES = ["image_flow_dmd2", "text_masked_kd", "image_flow_flow_map", "text_masked_coupled_kd"]


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("safetensors", [False, True], ids=["pt", "safetensors"])
def test_bundle_round_trip_is_bit_identical(name, safetensors, teachers, tmp_path, monkeypatch):
    if safetensors:
        pytest.importorskip("safetensors")
    else:
        monkeypatch.setattr("omnidiff.distill.export._safetensors", lambda: None)
    cfg = distill_config(ROOT / "configs" / "distill" / f"{name}.yaml", teachers, tmp_path)
    ckpt = DistillTrainer(cfg).train()
    ref = Pipeline.from_checkpoint(ckpt, device="cpu")
    bundle = export_bundle(ckpt, tmp_path / "bundle")
    monkeypatch.undo()
    pipe = load_bundle(bundle, device="cpu")

    meta = json.loads((bundle / "sampler.json").read_text())
    assert all(k in meta for k in SAMPLER_REQUIRED)
    assert meta["schema_version"] == 1
    assert meta["distilled_from"]["method"]
    assert meta["timesteps"] == ref.c.sampling.timesteps
    assert (bundle / "model.json").exists() and (bundle / "README.md").exists()
    assert (bundle / ("model.safetensors" if safetensors else "model.pt")).exists()

    a, b = ref.generate(n=3, seed=5), pipe.generate(n=3, seed=5)
    assert torch.equal(a.raw, b.raw)
    if isinstance(a.samples, list):
        assert a.samples == b.samples
    else:
        assert np.array_equal(a.samples, b.samples)


def test_cli_export(teachers, tmp_path):
    from omnidiff.core.cli import main

    cfg = distill_config(ROOT / "configs" / "distill" / "image_flow_kd.yaml", teachers, tmp_path)
    ckpt = DistillTrainer(cfg).train()
    main(["export", "--ckpt", str(ckpt), "--out", str(tmp_path / "b")])
    assert (tmp_path / "b" / "sampler.json").exists()
