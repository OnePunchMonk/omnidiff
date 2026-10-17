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
    return tr.process.q_sample(b.x0, t, maskable=tr.method.maskable(b.cond)), t, b.cond


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


def test_dcd_mean_target_is_the_teacher_marginal_for_a_consistent_teacher(tr, monkeypatch):
    # A teacher whose prediction ignores x and t is exactly self-consistent: its multi-step chain marginal
    # equals its one-step marginal. Then the mean target over many teacher paths must equal that marginal.
    n, L = 8192, tr.shape[0]
    V = tr.process.log_probs(tr.student(torch.zeros(1, L, dtype=torch.long), torch.ones(1), {})).shape[-1]
    fixed = torch.randn(L, V)
    pred = lambda x, *a: fixed.expand(x.shape[0], L, V)  # noqa: E731
    monkeypatch.setattr(tr, "teacher_pred", pred)
    monkeypatch.setattr(tr.method, "target", pred)
    x_t = torch.full((n, L), tr.process.mask_id)
    t = torch.full((n,), 0.9)
    for sub_steps in (1, 2):
        monkeypatch.setattr(tr.method, "sub_steps", sub_steps)
        mean = tr.method.targets(tr, x_t, t, torch.full((n,), 0.3), {}).mean(0)
        expected = tr.process.log_probs(fixed).exp()
        assert (mean - expected).abs().max() < 0.03


def test_dcd_maskable_respects_prefix_and_loss_mask(tr):
    prefix = torch.tensor([[5, -1, -1, -1]])
    loss_mask = torch.tensor([[True, True, True, False]])
    m = tr.method.maskable
    assert m({}) is None
    assert m({"prefix": prefix}).tolist() == [[False, True, True, True]]
    assert m({"loss_mask": loss_mask}).tolist() == [[True, True, True, False]]
    assert m({"prefix": prefix, "loss_mask": loss_mask}).tolist() == [[False, True, True, False]]


def test_dcd_teacher_steps_go_through_teacher_pred(tr, monkeypatch):
    # the toy text teacher has no CFG branch, so guidance cannot change its output; check the call path
    calls = []
    orig = tr.teacher_pred
    monkeypatch.setattr(tr, "teacher_pred", lambda *a: calls.append(1) or orig(*a))
    x_t, t, cond = _state(tr, 0.8)
    tr.method.targets(tr, x_t, t, torch.full((4,), 0.4), cond)
    assert len(calls) == tr.method.sub_steps


def test_dcd_target_ema_works_with_lora(tmp_path_factory):
    teacher = SFTTrainer(tiny_config(ROOT / "configs" / "text_masked_grpo.yaml", tmp_path_factory.mktemp("t"))).train()
    cfg = load_config(ROOT / "configs" / "distill" / "text_masked_dcd.yaml",
                      ["device=cpu", f"output_dir={tmp_path_factory.mktemp('o')}", *TINY, "distill.lora={rank: 2, alpha: 2}"])
    cfg["teacher"] = str(teacher)
    tr = DistillTrainer(cfg)
    assert set(tr.method.target.state_dict()) == set(tr.student.state_dict())
    tr.train()
    state, target = tr.student.state_dict(), tr.method.target.state_dict()
    # the target follows the trainable LoRA weights, lagging them
    assert any("lora" in k.lower() and not torch.equal(target[k], state[k]) for k in target)
