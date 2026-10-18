from pathlib import Path

import numpy as np
from diffusion_omni.core.testing import tiny_config
from diffusion_omni.posttrain import SFTTrainer
from diffusion_omni.reward.metrics import (
    frechet_distance,
    precision_recall,
    within_prompt_diversity,
)
from diffusion_omni.reward.scorecard import run_scorecard, to_markdown

ROOT = Path(__file__).resolve().parents[3]


def test_frechet_distance_matches_closed_form_for_gaussians():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (20000, 3))
    b = rng.normal(0, 1, (20000, 3)) * np.array([1.0, 2.0, 1.0]) + np.array([1.0, 0.0, 0.0])
    # ||mu||^2 + tr(S1 + S2 - 2 (S1 S2)^1/2) with S1 = I, S2 = diag(1, 4, 1): 1 + (6 + 3 - 2 * 4) = 2
    assert abs(frechet_distance(a, b) - 2.0) < 0.1
    assert abs(frechet_distance(a, a)) < 1e-6


def test_precision_recall_limits():
    rng = np.random.default_rng(0)
    real = rng.normal(0, 1, (300, 2))
    assert precision_recall(real, real) == (1.0, 1.0)
    p, r = precision_recall(real, real + 100)
    assert p == 0.0 and r == 0.0
    collapsed = np.repeat(real[:1], 300, axis=0) + rng.normal(0, 1e-3, (300, 2))
    p, r = precision_recall(real, collapsed)
    assert p == 1.0 and r < 0.1  # mode collapse: precise, no coverage


def test_within_prompt_diversity():
    assert within_prompt_diversity(["a", "a", "b", "c"], [0, 0, 1, 1]) == 0.75
    x = np.zeros((4, 3))
    x[1] = 1.0
    assert within_prompt_diversity(x, [0, 0, 1, 1]) > 0 and within_prompt_diversity(x[[0, 0]], [0, 0]) == 0


def test_scorecard_runs_on_a_tiny_teacher(tmp_path):
    ckpt = SFTTrainer(tiny_config(ROOT / "configs" / "image_flow_grpo.yaml", tmp_path)).train()
    rows = run_scorecard(str(ckpt), {"same": str(ckpt)}, nfes=(1, 2), n=16, ref_steps=4,
                         variants={"shift3": ["process.shift=3.0"]})
    assert [(r["model"], r["nfe"]) for r in rows] == [
        ("reference@4", 4), ("reference", 1), ("reference", 2), ("reference[shift3]", 1), ("reference[shift3]", 2),
        ("same", 1), ("same", 2)]
    assert {"fd", "precision", "recall", "data_fd", "class_acc", "reward"} <= set(rows[1])
    assert "| model |" in to_markdown(rows)
