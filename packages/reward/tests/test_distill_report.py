"""The multi-seed orchestration in scripts/distill_toy_report.py, on a tiny teacher and one student."""

import importlib.util
import json
from pathlib import Path

import pytest

from diffusion_omni.core.testing import tiny_config
from diffusion_omni.posttrain import SFTTrainer

ROOT = Path(__file__).resolve().parents[3]
TINY_STUDENT = ["distill.steps=2", "distill.pool_size=8", "distill.batch_size=4", "distill.eval_every=100"]


@pytest.fixture(scope="module")
def report():
    spec = importlib.util.spec_from_file_location("distill_toy_report", ROOT / "scripts" / "distill_toy_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env(report, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(report, "GROUPS", {"image_flow_grpo": ["image_flow_kd"]})
    monkeypatch.setattr(report, "VARIANTS", {})
    SFTTrainer(tiny_config(ROOT / "configs" / "image_flow_grpo.yaml", "runs/image_flow_grpo")).train()
    return tmp_path


def run(report, *extra):
    args = report.build_parser().parse_args(["--seeds", "0", "1", "-n", "8", "--nfe", "1", "--out", "report.md", *extra])
    return report.main_seeds(args)


def test_dry_run_creates_nothing(report, env, capsys):
    before = sorted(p for p in env.rglob("*"))
    run(report, "--dry-run", "--set", *TINY_STUDENT)
    assert sorted(p for p in env.rglob("*")) == before
    assert "2 to train" in capsys.readouterr().out


def test_two_seeds_provenance_and_stale_detection(report, env):
    run(report, "--set", *TINY_STUDENT)
    runs = sorted(p.name for p in (env / "runs/distill_smoke").iterdir())
    assert runs == ["image_flow_kd_s0", "image_flow_kd_s1"]
    for r in runs:
        prov = json.loads((env / "runs/distill_smoke" / r / "provenance.json").read_text())
        assert prov["cfg"]["seed"] == int(r[-1]) and prov["extra"] == TINY_STUDENT
        assert (env / "runs/distill_smoke" / r / "distill/final/model.pt").exists()
    assert not (env / "runs/distill").exists()  # --set runs stay out of the real root
    md = (env / "report.md").read_text()
    assert "±" in md and "sample std" in md
    rows = json.loads((env / "runs/scorecards/image_flow_grpo_seeds.json").read_text())
    assert {r.get("seed") for r in rows if r["model"] == "image_flow_kd"} == {0, 1}

    stamp = (env / "runs/distill_smoke/image_flow_kd_s0/distill/final/model.pt").stat().st_mtime_ns
    run(report, "--set", *TINY_STUDENT)  # unchanged: nothing is retrained
    assert (env / "runs/distill_smoke/image_flow_kd_s0/distill/final/model.pt").stat().st_mtime_ns == stamp

    changed = [*TINY_STUDENT[:-1], "distill.eval_every=50"]
    with pytest.raises(SystemExit, match="stale"):
        run(report, "--set", *changed)
    run(report, "--retrain-stale", "--set", *changed)
    prov = json.loads((env / "runs/distill_smoke/image_flow_kd_s0/provenance.json").read_text())
    assert prov["extra"] == changed


def test_missing_teacher_fails_unless_allowed(report, env):
    with pytest.raises(SystemExit, match="missing teachers"):
        run(report, "--teacher-suffix", "_ts10", "--set", *TINY_STUDENT)
    run(report, "--teacher-suffix", "_ts10", "--allow-missing", "--dry-run", "--set", *TINY_STUDENT)


def test_legacy_reuse_refuses_set_extras_and_missing_runs(report, env):
    args = report.build_parser().parse_args(["--seeds", "0", "--reuse-seed0"])
    teacher = Path("runs/image_flow_grpo/sft/final")
    cfg = report.student_cfg("image_flow_kd", teacher, 0, Path("x"), [])
    assert "extras" in report.legacy_problem("image_flow_kd", cfg, teacher, ["a=1"], Path("runs/distill/image_flow_kd"))
    assert report.legacy_problem("image_flow_kd", cfg, teacher, [], Path("runs/distill/image_flow_kd")) == "no model.pt"
    assert args.reuse_seed0
