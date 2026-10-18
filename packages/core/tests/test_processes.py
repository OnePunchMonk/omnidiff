import math

import pytest
import torch
from diffusion_omni.core.processes.flow import FlowMatching
from diffusion_omni.core.processes.gaussian import GaussianDiffusion
from diffusion_omni.core.processes.masked import MaskedDiffusion

CONTINUOUS = [
    GaussianDiffusion(eta=1.0, reduction="sum"),
    GaussianDiffusion(schedule="cosine", eta=0.5, reduction="sum"),
    FlowMatching(noise_level=0.7, reduction="sum"),
    FlowMatching(noise_level=0.5, shift=3.0, reduction="sum"),
]


@pytest.mark.parametrize("proc", CONTINUOUS, ids=lambda p: type(p).__name__)
def test_continuous_log_prob_matches_torch_normal(proc):
    torch.manual_seed(0)
    x, pred = torch.randn(3, 2, 4, 4), torch.randn(3, 2, 4, 4)
    ts = proc.timesteps(5, "cpu")
    for k in range(4):  # last step of DDIM is deterministic, check the others
        tr = proc.transition(pred, x, ts[k], ts[k + 1])
        ref = torch.distributions.Normal(tr.stats["mean"], tr.stats["std"]).log_prob(tr.x_next).sum((1, 2, 3))
        torch.testing.assert_close(tr.log_prob, ref)
        # rescoring the sampled action reproduces its log-prob
        torch.testing.assert_close(proc.transition(pred, x, ts[k], ts[k + 1], x_next=tr.x_next).log_prob, tr.log_prob)
        assert torch.all(proc.kl(tr, tr) == 0)


def test_deterministic_samplers_have_no_log_prob():
    x = pred = torch.randn(2, 3, 4, 4)
    t, s = torch.tensor(0.5), torch.tensor(0.25)
    assert GaussianDiffusion(eta=0.0).transition(pred, x, t, s).log_prob is None
    assert FlowMatching(noise_level=0.0).transition(pred, x, t, s).log_prob is None
    # DDPM's final step to t=0 is deterministic too
    assert GaussianDiffusion(eta=1.0).transition(pred, x, s, torch.tensor(0.0)).log_prob is None


def test_flow_ode_step_is_euler():
    x, v = torch.randn(2, 3), torch.randn(2, 3)
    tr = FlowMatching(noise_level=0.0).transition(v, x, torch.tensor(0.8), torch.tensor(0.6))
    torch.testing.assert_close(tr.x_next, x - 0.2 * v)


def test_flow_sde_is_marginal_preserving_on_gaussian_data():
    # With x0 ~ N(0, s^2), p_t is Gaussian and the exact velocity is known in closed form,
    # so integrating the SDE from pure noise must land back on N(0, s^2).
    s = 0.5
    proc = FlowMatching(noise_level=0.7)

    def velocity(x, t):
        var = (1 - t) ** 2 * s**2 + t**2
        return x * (t - (1 - t) * s**2) / var  # E[noise - x0 | x_t]

    torch.manual_seed(0)
    ts = proc.timesteps(200, "cpu")
    x = torch.randn(20000)
    for k in range(200):
        x = proc.transition(velocity(x, ts[k]), x, ts[k], ts[k + 1]).x_next
    assert abs(x.std().item() - s) < 0.03


def _masked(seed=0, V=7, B=4, L=10):
    torch.manual_seed(seed)
    proc = MaskedDiffusion(mask_id=V - 1)
    prefix = torch.full((B, L), -1)
    prefix[:, : min(3, L)] = torch.tensor([1, 2, 3])[:L]
    return proc, prefix, V


def test_masked_process_decodes_everything_and_keeps_prefix():
    proc, prefix, V = _masked()
    x = proc.prior(prefix.shape, {"prefix": prefix}, "cpu")
    ts = proc.timesteps(6, "cpu")
    for k in range(6):
        logits = torch.randn(*x.shape, V)
        tr = proc.transition(logits, x, ts[k], ts[k + 1])
        decoded = x != proc.mask_id
        assert torch.equal(tr.x_next[decoded], x[decoded])  # carry-over
        torch.testing.assert_close(proc.transition(logits, x, ts[k], ts[k + 1], x_next=tr.x_next).log_prob, tr.log_prob)
        assert torch.allclose(proc.kl(tr, tr), torch.zeros(x.shape[0]))
        x = tr.x_next
    assert not (x == proc.mask_id).any()
    assert torch.equal(x[:, :3], prefix[:, :3])


def test_masked_log_prob_is_exact():
    proc, _, V = _masked(V=5, B=1, L=2)
    logits = torch.randn(1, 2, V)
    x_t = torch.full((1, 2), proc.mask_id)
    x_next = torch.tensor([[2, proc.mask_id]])  # first unmasked to 2, second stays masked
    t, s = torch.tensor(0.75), torch.tensor(0.25)
    tr = proc.transition(logits, x_t, t, s, x_next=x_next)
    logp = logits[0, 0, : V - 1].log_softmax(-1)[2]
    expected = math.log(0.5 / 0.75) + logp + math.log(0.25 / 0.75)
    torch.testing.assert_close(tr.log_prob, expected.reshape(1))


def test_masked_kl_is_differentiable_and_positive():
    proc, prefix, V = _masked()
    x = proc.prior(prefix.shape, {"prefix": prefix}, "cpu")
    a = torch.randn(*x.shape, V, requires_grad=True)
    t, s = torch.tensor(1.0), torch.tensor(0.5)
    new = proc.transition(a, x, t, s)
    ref = proc.transition(torch.randn(*x.shape, V), x, t, s, x_next=new.x_next)
    kl = proc.kl(new, ref)
    assert (kl > 0).all()
    kl.sum().backward()
    assert torch.isfinite(a.grad).all()
