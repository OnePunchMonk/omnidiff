from pathlib import Path

import pytest
import torch

from omnidiff.core.config import load_config
from omnidiff.core.testing import tiny_config
from omnidiff.distill import DistillTrainer
from omnidiff.posttrain import SFTTrainer

ROOT = Path(__file__).resolve().parents[3]
TINY = ["distill.steps=2", "distill.batch_size=4", "distill.pool_size=8", "distill.teacher_steps=4",
        "distill.log_every=1", "distill.eval_every=0"]


@pytest.fixture(scope="module")
def tr(tmp_path_factory):
    teacher = SFTTrainer(tiny_config(ROOT / "configs" / "text_masked_grpo.yaml", tmp_path_factory.mktemp("t"))).train()
    cfg = load_config(ROOT / "configs" / "distill" / "text_masked_dcd.yaml",
                      ["device=cpu", f"output_dir={tmp_path_factory.mktemp('o')}", *TINY])
    cfg["teacher"] = str(teacher)
    return DistillTrainer(cfg)


def _state(tr, t):
    b = tr.data.batch(4)
    t = torch.full((4,), t)
    return tr.process.q_sample(b.x0, t), t, b.cond


def test_dcd_targets_are_distributions_and_hit_the_boundary(tr):
    m, mask = tr.method, tr.process.mask_id
    x_t, t, cond = _state(tr, 0.8)
    masked = x_t == mask
    for s in (0.5, 0.0):
        target = m.targets(tr, x_t, t, torch.full((4,), s), cond)
        torch.testing.assert_close(target[masked].sum(-1), torch.ones(int(masked.sum())))
        assert (target >= 0).all() and (target[..., mask] == 0).all()
    # s = 0: the teacher has decoded everything, so every masked position is a point mass
    assert torch.allclose(target[masked].max(-1).values, torch.ones(int(masked.sum())))


def test_dcd_fixed_point_zero_loss_when_student_equals_target_at_s_equals_t(tr):
    m = tr.method
    x_t, t, cond = _state(tr, 0.6)
    masked = x_t == tr.process.mask_id
    target = m.targets(tr, x_t, t, t.clone(), cond)  # no teacher step: x_s = x_t
    with torch.no_grad():
        logp = tr.process.log_probs(tr.student(x_t, t, cond))
    assert torch.allclose(target[masked], logp.exp()[masked], atol=1e-5)  # student starts as the target
    assert m.kl(target, logp)[masked].abs().max() < 1e-5
    # perturbing the student away from its target makes the loss positive
    with torch.no_grad():
        logp2 = tr.process.log_probs(tr.student(x_t, t, cond) + torch.randn(*logp.shape))
    assert m.kl(target, logp2)[masked].mean() > 1e-3
