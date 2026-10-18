import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusion_omni.core import Pipeline
from diffusion_omni.core.config import load_config, save_config
from diffusion_omni.core.testing import tiny_config
from diffusion_omni.distill import DistillTrainer
from diffusion_omni.distill.export import SAMPLER_REQUIRED, export_bundle, load_bundle
from diffusion_omni.posttrain import SFTTrainer

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
        monkeypatch.setattr("diffusion_omni.distill.export._safetensors", lambda: None)
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


def _student(teachers, tmp_path, name="image_flow_kd"):
    cfg = distill_config(ROOT / "configs" / "distill" / f"{name}.yaml", teachers, tmp_path)
    return DistillTrainer(cfg).train()


def test_cli_export_loads_and_matches(teachers, tmp_path):
    from diffusion_omni.core.cli import main

    ckpt = _student(teachers, tmp_path)
    main(["export", "--ckpt", str(ckpt), "--out", str(tmp_path / "b")])
    a = Pipeline.from_checkpoint(ckpt, device="cpu").generate(n=2, seed=1)
    b = load_bundle(tmp_path / "b", device="cpu").generate(n=2, seed=1)
    assert torch.equal(a.raw, b.raw)


def test_plain_sft_checkpoint_round_trips(teachers, tmp_path):
    ckpt = teachers("image_flow_grpo")
    bundle = export_bundle(ckpt, tmp_path / "sft")
    assert "distilled_from" not in json.loads((bundle / "sampler.json").read_text())
    a = Pipeline.from_checkpoint(ckpt, device="cpu").generate(n=2, seed=3)
    assert torch.equal(a.raw, load_bundle(bundle, device="cpu").generate(n=2, seed=3).raw)


def test_provenance_has_no_absolute_teacher_path(teachers, tmp_path):
    bundle = export_bundle(_student(teachers, tmp_path), tmp_path / "b")
    teacher = json.loads((bundle / "sampler.json").read_text())["distilled_from"]["teacher"]
    assert not Path(teacher).is_absolute()


def test_re_export_does_not_load_stale_weights(teachers, tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    a, b = teachers("image_flow_grpo"), _student(teachers, tmp_path)
    out = tmp_path / "bundle"
    export_bundle(a, out)
    monkeypatch.setattr("diffusion_omni.distill.export._safetensors", lambda: None)
    export_bundle(b, out)
    monkeypatch.undo()
    assert not (out / "model.safetensors").exists()
    ref = Pipeline.from_checkpoint(b, device="cpu").generate(n=2, seed=0)
    assert torch.equal(ref.raw, load_bundle(out, device="cpu").generate(n=2, seed=0).raw)


def test_tampered_weights_are_rejected(teachers, tmp_path):
    bundle = export_bundle(teachers("image_flow_grpo"), tmp_path / "b")
    meta = json.loads((bundle / "sampler.json").read_text())
    with open(bundle / meta["weights"]["file"], "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="sha256"):
        load_bundle(bundle)


def test_relocated_bundle_resolves_resource_paths(teachers, tmp_path):
    src = tmp_path / "ckpt"
    shutil.copytree(teachers("text_masked_grpo"), src)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("the cat\na dog\n")
    cfg = load_config(src / "config.yaml")
    cfg["prompts"] = {"type": "text", "path": str(prompts)}
    save_config(cfg, src / "config.yaml")
    bundle = export_bundle(src, tmp_path / "bundle")
    ref = Pipeline.from_checkpoint(src, device="cpu").generate(n=2, seed=0)
    prompts.unlink()
    moved = shutil.move(str(bundle), str(tmp_path / "elsewhere"))
    assert load_bundle(moved, device="cpu").generate(n=2, seed=0).samples == ref.samples


@pytest.fixture
def good_meta(teachers, tmp_path):
    bundle = export_bundle(_student(teachers, tmp_path), tmp_path / "b")
    return bundle, json.loads((bundle / "sampler.json").read_text())


@pytest.mark.parametrize("edit, match", [
    (lambda m: m.pop("weights"), "missing keys"),
    (lambda m: m.update(schema_version=2), "schema_version"),
    (lambda m: m.update(process={"settings": {}}), "process"),
    (lambda m: m.update(timesteps=[1.0, 0.2, 0.5, 0.0], num_steps=3), "descending"),
    (lambda m: m.update(timesteps=[0.9, 0.5, 0.0], num_steps=2), "from 1 to 0"),
    (lambda m: m.update(timesteps=["1", "0"], num_steps=1), "descending"),
    (lambda m: m.update(num_steps=m["num_steps"] + 1), "num_steps"),
    (lambda m: m.update(sample_shape=[]), "sample_shape"),
    (lambda m: m.update(weights={"file": "model.pt"}), "weights"),
])
def test_malformed_sampler_json_is_rejected(good_meta, edit, match):
    bundle, meta = good_meta
    edit(meta)
    (bundle / "sampler.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=match):
        load_bundle(bundle)


def test_export_refuses_to_overwrite_a_checkpoint(teachers):
    ckpt = teachers("image_flow_grpo")
    with pytest.raises(ValueError, match="checkpoint"):
        export_bundle(ckpt, ckpt)
    assert (Path(ckpt) / "model.pt").exists()
