from pathlib import Path

import numpy as np
import pytest
import torch

from diffusion_omni.core.config import load_config
from diffusion_omni.core.registry import REWARDS
from diffusion_omni.core.testing import tiny_config
from diffusion_omni.distill import DistillTrainer
from diffusion_omni.posttrain import SFTTrainer
from diffusion_omni.reward import rewards  # noqa: F401

ROOT = Path(__file__).resolve().parents[3]
TINY = ["device=cpu", "distill.steps=2", "distill.batch_size=4", "distill.pool_size=8", "distill.teacher_steps=4"]


@pytest.fixture(scope="module")
def teacher(tmp_path_factory):
    return SFTTrainer(tiny_config(ROOT / "configs" / "image_flow_grpo.yaml", tmp_path_factory.mktemp("t"))).train()


def _trainer(teacher, tmp_path, *extra):
    cfg = load_config(ROOT / "configs" / "distill" / "image_flow_dmd2_reward.yaml", [*TINY, f"output_dir={tmp_path}", *extra])
    cfg["teacher"] = str(teacher)
    return DistillTrainer(cfg)


def _fixed_batch(tr, m, n=4):
    """Pin the noise level and noise so the generator loss is a deterministic function of x0."""
    torch.manual_seed(0)
    with torch.no_grad():  # make the fake score differ from the teacher
        for p in m.fake["score"].parameters():
            p.add_(0.02 * torch.randn_like(p))
    tau, eps = torch.full((n,), 0.5), torch.randn(n, *tr.shape)
    x0, cond = 0.3 * torch.randn(n, *tr.shape), tr.prompt_cond(n)
    m._noisy = lambda tr_, x: (tr_.process.q_sample(x, tau, eps), tau)
    return x0, cond


def _grad(loss_fn, tr, x0, cond):
    x = x0.clone().requires_grad_(True)
    loss = loss_fn(tr, x, cond)[0]
    loss.backward()
    return loss.detach(), x.grad


@pytest.mark.parametrize("mode", ["gradient", "advantage"])
def test_zero_reward_weight_reproduces_dmd2(mode, teacher, tmp_path):
    tr = _trainer(teacher, tmp_path, f"distill.method.mode={mode}", "distill.method.reward_weight=0")
    m = tr.method
    x0, cond = _fixed_batch(tr, m)
    loss, grad = _grad(m.generator_loss, tr, x0, cond)
    ref_loss, ref_grad = _grad(super(type(m), m).generator_loss, tr, x0, cond)
    assert torch.equal(loss, ref_loss)
    assert torch.equal(grad, ref_grad)


def test_reward_gradient_matches_autograd_and_closed_form(teacher, tmp_path):
    tr = _trainer(teacher, tmp_path, "distill.method.reward_weight=0.7")
    m = tr.method
    x0, cond = _fixed_batch(tr, m)
    _, grad = _grad(m.generator_loss, tr, x0, cond)
    _, dmd2_grad = _grad(super(type(m), m).generator_loss, tr, x0, cond)
    # closed form for channel-0 dominance on decode(x0) = (clamp(x0) + 1) / 2, mean over the batch:
    # dr_i/dx0 = 0.5 * [1/(HW) on channel 0, -1/((C-1)HW) on the others] inside the clamp range
    b, c, h, w = x0.shape
    coef = torch.full((c,), -1.0 / ((c - 1) * h * w))
    coef[0] = 1.0 / (h * w)
    d_reward = 0.5 * coef.view(1, c, 1, 1) * (x0.abs() < 1).float() / b
    torch.testing.assert_close(grad, dmd2_grad - 0.7 * d_reward)

    def combined(tr_, x, cond_):  # the combined loss written out with plain autograd
        d = (x.clamp(-1, 1) + 1) / 2
        r = d[:, 0].flatten(1).mean(1) - d[:, 1:].flatten(1).mean(1)
        return (super(type(m), m).generator_loss(tr_, x, cond_)[0] - 0.7 * r.mean(),)

    torch.testing.assert_close(grad, _grad(combined, tr, x0, cond)[1])


def test_advantage_weights_have_group_mean_one_and_follow_reward(teacher, tmp_path):
    tr = _trainer(teacher, tmp_path, "distill.method.mode=advantage", "distill.method.reward_weight=2.0", "distill.method.group_size=3")
    m = tr.method
    x0, cond = _fixed_batch(tr, m, n=6)
    _, g_adv = _grad(m.generator_loss, tr, x0, cond)
    _, g_plain = _grad(super(type(m), m).generator_loss, tr, x0, cond)
    ratio = (g_adv.flatten(1) / g_plain.flatten(1)).mean(1)  # per-sample gradient scale (constant within a sample)
    r = torch.as_tensor(m.reward(tr.c.codec.decode(x0), [""] * 6))
    ratio, r = ratio.view(2, 3), r.view(2, 3)  # two groups of three
    torch.testing.assert_close(ratio.mean(1), torch.ones(2), atol=1e-4, rtol=0)
    assert torch.equal(ratio.argsort(1), r.argsort(1))


@pytest.mark.parametrize("spec", [{"type": "channel_dominance", "channel": 1}, {"type": "brightness"},
                                  {"type": "brightness", "target": 0.4}, {"type": "temporal_smoothness"},
                                  {"type": "composite", "terms": [{"weight": 2.0, "reward": {"type": "brightness"}},
                                                                  {"weight": 1.0, "reward": {"type": "channel_dominance"}}]}])
def test_torch_reward_matches_numpy(spec):
    rw = REWARDS.build(spec)
    x = np.random.default_rng(0).random((5, 3, 4, 6, 6)).astype(np.float32)  # (B, C, T, H, W)
    if spec["type"] != "temporal_smoothness":
        x = x[:, :, 0]
    np.testing.assert_allclose(rw.torch_call(torch.as_tensor(x)).numpy(), rw(x, [""] * 5), rtol=1e-5, atol=1e-6)


def test_decode_tensor_matches_decode(teacher, tmp_path):
    tr = _trainer(teacher, tmp_path)
    x = 1.5 * torch.randn(3, *tr.shape)
    np.testing.assert_allclose(tr.c.codec.decode_tensor(x).numpy(), tr.c.codec.decode(x))


@pytest.mark.parametrize("mode", ["gradient", "advantage"])
def test_zero_reward_weight_student_loss_equals_dmd2(mode, teacher, tmp_path):
    """At the loss("student") level: same prompts, same rng consumption, same loss."""
    import copy

    from diffusion_omni.distill.methods.dmd2 import DMD2
    tr = _trainer(teacher, tmp_path, f"distill.method.mode={mode}", "distill.method.reward_weight=0")
    m = tr.method
    rng, gen = copy.deepcopy(tr.rng), tr.gen.get_state()
    torch.manual_seed(1)
    a, _ = m.loss("student", tr)
    tr.rng = rng
    tr.gen.set_state(gen)
    torch.manual_seed(1)
    b, _ = DMD2.loss(m, "student", tr)
    assert torch.equal(a, b)
