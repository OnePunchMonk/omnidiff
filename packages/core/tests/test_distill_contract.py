"""The parts of core that few-step distillation builds on."""

import pytest
import torch
from torch.func import jvp

from omnidiff.core.nn.denoiser import build_model
from omnidiff.core.nn.layers import math_attention
from omnidiff.core.processes.flow import FlowMatching
from omnidiff.core.processes.gaussian import GaussianDiffusion
from omnidiff.core.processes.masked import MaskedDiffusion
from omnidiff.core.sampling import predict, sample


def _image_model(two_time=False, num_classes=3):
    torch.manual_seed(0)
    return build_model({"dim": 32, "adapter": {"type": "image_patch", "patch": 2},
                        "backbone": {"type": "dit", "depth": 1, "heads": 2},
                        "conditioner": {"type": "class", "num_classes": num_classes}, "two_time": two_time},
                       sample_shape=(2, 4, 4))


@pytest.mark.parametrize("proc", [GaussianDiffusion(), FlowMatching()], ids=["gaussian", "flow"])
def test_parameterisation_roundtrip(proc):
    torch.manual_seed(0)
    x0, noise, t = torch.randn(4, 2, 4, 4), torch.randn(4, 2, 4, 4), torch.rand(4) * 0.9 + 0.05
    x_t = proc.q_sample(x0, t, noise)
    pred = noise if isinstance(proc, GaussianDiffusion) else noise - x0  # the exact model output
    torch.testing.assert_close(proc.to_x0(pred, x_t, t), x0, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(proc.to_eps(pred, x_t, t), noise, atol=1e-4, rtol=1e-4)
    # with the exact output, the ODE step lands exactly on the same (x0, noise) path
    s = t * 0.5
    torch.testing.assert_close(proc.ode_step(pred, x_t, t, s), proc.q_sample(x0, s, noise), atol=1e-4, rtol=1e-4)


def test_ode_step_matches_existing_samplers():
    x, pred = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
    t, s = torch.tensor(0.7), torch.tensor(0.4)
    flow = FlowMatching(noise_level=0.0)
    torch.testing.assert_close(flow.ode_step(pred, x, t, s), x + (s - t) * pred)
    gauss = GaussianDiffusion(eta=0.0)
    torch.testing.assert_close(gauss.ode_step(pred, x, t, s), gauss.transition(pred, x, t, s).x_next)


@pytest.mark.parametrize("proc", [GaussianDiffusion(sampler="renoise", reduction="sum"),
                                  FlowMatching(sampler="renoise", reduction="sum")], ids=["gaussian", "flow"])
def test_renoise_is_an_exact_gaussian_policy(proc):
    torch.manual_seed(0)
    x, pred = torch.randn(3, 2, 4, 4), torch.randn(3, 2, 4, 4)
    t, s = torch.tensor(0.8), torch.tensor(0.3)
    tr = proc.transition(pred, x, t, s)
    a, std = proc.alpha_sigma(s)
    mean = a * proc.to_x0(pred, x, t)
    ref = torch.distributions.Normal(mean, std).log_prob(tr.x_next).sum((1, 2, 3))
    torch.testing.assert_close(tr.log_prob, ref)
    torch.testing.assert_close(proc.transition(pred, x, t, s, x_next=tr.x_next).log_prob, tr.log_prob)
    final = proc.transition(pred, x, s, torch.tensor(0.0))
    assert final.log_prob is None
    torch.testing.assert_close(final.x_next, proc.to_x0(pred, x, s))


def test_masked_q_sample_rate_and_keep_mask():
    proc = MaskedDiffusion(mask_id=9)
    x0 = torch.randint(0, 9, (64, 200))
    keep = torch.zeros_like(x0, dtype=torch.bool)
    keep[:, 100:] = True
    x_t = proc.q_sample(x0, 0.3, maskable=keep)
    assert not (x_t[:, :100] == 9).any()
    assert abs((x_t[:, 100:] == 9).float().mean().item() - 0.3) < 0.02
    per_sample = proc.q_sample(x0, torch.tensor([0.0, 1.0]).repeat(32))
    assert not (per_sample[0::2] == 9).any() and (per_sample[1::2] == 9).all()


def test_two_time_model_starts_as_the_one_time_model():
    one, two = _image_model(), _image_model(two_time=True)
    missing, unexpected = two.load_state_dict(one.state_dict(), strict=False)
    assert all(k.startswith("s_embed.") for k in missing) and not unexpected
    with torch.no_grad():  # move off the adaLN-Zero init, but keep s_embed's zero output layer
        for name, p in two.named_parameters():
            if not name.startswith("s_embed.mlp.2."):
                p.add_(0.05 * torch.randn_like(p))
    one.load_state_dict({k: v for k, v in two.state_dict().items() if not k.startswith("s_embed.")})
    x, cond = torch.randn(2, 2, 4, 4), {"class_ids": torch.tensor([0, 1])}
    t = torch.tensor([0.9, 0.6])
    torch.testing.assert_close(two(x, t, cond, s=t * 0.1), one(x, t, cond))


def test_math_attention_matches_sdpa_and_supports_jvp():
    model = _image_model(two_time=True)
    x, t, cond = torch.randn(2, 2, 4, 4), torch.tensor([0.8, 0.5]), {"class_ids": torch.tensor([0, 2])}
    fused = model(x, t, cond, s=t / 2)
    with math_attention():
        torch.testing.assert_close(model(x, t, cond, s=t / 2), fused, atol=1e-5, rtol=1e-5)
        v = torch.randn_like(x)
        out, tangent = jvp(lambda z, tt: model(z, tt, cond, s=t / 2), (x, t), (v, torch.ones_like(t)))
    eps = 1e-3
    fd = (model(x + eps * v, t + eps, cond, s=t / 2) - model(x - eps * v, t - eps, cond, s=t / 2)) / (2 * eps)
    torch.testing.assert_close(tangent, fd, atol=2e-3, rtol=2e-2)


def test_guidance_with_per_sample_times_matches_per_sample_calls():
    model, proc = _image_model(), FlowMatching()
    x, cond = torch.randn(3, 2, 4, 4), {"class_ids": torch.tensor([0, 1, 2])}
    t = torch.tensor([0.9, 0.5, 0.2])
    batched = predict(model, x, t, cond, 3.0, proc)
    for i in range(3):
        single = predict(model, x[i:i + 1], t[i], {"class_ids": cond["class_ids"][i:i + 1]}, 3.0, proc)
        torch.testing.assert_close(batched[i:i + 1], single, atol=1e-5, rtol=1e-5)


def test_sample_with_explicit_timesteps():
    model, proc = _image_model(), FlowMatching(sampler="renoise")
    traj = sample(model, proc, (2, 2, 4, 4), {"class_ids": torch.tensor([0, 1])}, timesteps=[1.0, 0.6, 0.0],
                  keep_trajectory=True)
    assert traj.xs.shape[1] == 3 and traj.stochastic.tolist() == [True, False]
    with pytest.raises(ValueError, match="descending"):
        sample(model, proc, (1, 2, 4, 4), {"class_ids": torch.tensor([0])}, timesteps=[0.0, 1.0])


def test_renoise_asks_two_time_models_for_the_jump_to_data():
    model, proc = _image_model(two_time=True), FlowMatching(sampler="renoise")
    seen = []
    orig = model.forward
    model.forward = lambda x, t, cond=None, drop_cond=None, s=None: seen.append(float(s)) or orig(x, t, cond, drop_cond, s)
    sample(model, proc, (1, 2, 4, 4), {"class_ids": torch.tensor([0])}, timesteps=[1.0, 0.5, 0.0])
    assert seen == [0.0, 0.0]
    seen.clear()
    sample(model, FlowMatching(sampler="ode"), (1, 2, 4, 4), {"class_ids": torch.tensor([0])}, timesteps=[1.0, 0.5, 0.0])
    assert seen == [0.5, 0.0]
