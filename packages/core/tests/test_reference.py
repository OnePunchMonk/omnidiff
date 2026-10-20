import pytest
import torch
from diffusion_omni.core import Pipeline, parity, reference_sample, sample
from diffusion_omni.core.processes.flow import FlowMatching
from diffusion_omni.core.processes.gaussian import GaussianDiffusion
from diffusion_omni.core.processes.masked import MaskedDiffusion
from diffusion_omni.core.testing import tiny_config

MU, S = 0.3, 0.5  # toy data x0 ~ N(MU, S^2) per element


class GaussianOracle(torch.nn.Module):
    """The exact denoiser for Gaussian data: E[noise - x0 | x_t] for flow, E[noise | x_t] for VP."""

    def __init__(self, process):
        super().__init__()
        self.process = process
        self.unused = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, t, cond=None, drop_cond=None):
        a, s = self.process.alpha_sigma(t.to(x.dtype))
        eps = s * (x - a * MU) / (a**2 * S**2 + s**2)
        if isinstance(self.process, GaussianDiffusion):
            return eps
        x0 = (x - s * eps) / a.clamp(min=1e-12)
        return torch.where(t >= 1, eps - MU, eps - x0)  # at t = 1, x0_hat = MU


def _transport(process, z):
    # Both ODEs are monotone maps between Gaussians, so the endpoint is known per element.
    a, s = process.alpha_sigma(torch.tensor(1.0, dtype=torch.float64))
    return MU + S * (z - a * MU) / (a**2 * S**2 + s**2).sqrt()


@pytest.mark.parametrize("proc", [FlowMatching(), FlowMatching(shift=3.0), GaussianDiffusion(),
                                  GaussianDiffusion(schedule="cosine")], ids=lambda p: f"{type(p).__name__}")
def test_reference_lands_on_analytic_transport(proc):
    z = torch.randn(64, 8, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    x = reference_sample(GaussianOracle(proc), proc, z.shape, num_steps=400, x_init=z)
    assert x.dtype == torch.float64
    torch.testing.assert_close(x, _transport(proc, z), atol=2e-3, rtol=0)


def test_heun_beats_euler_at_equal_steps():
    proc = FlowMatching(sampler="ode")
    oracle = GaussianOracle(proc).double()
    z = torch.randn(64, 8, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    target = _transport(proc, z)
    heun = (reference_sample(oracle, proc, z.shape, num_steps=20, x_init=z) - target).abs().max()
    x = z
    for t, t_next in zip(proc.timesteps(20, "cpu")[:-1], proc.timesteps(20, "cpu")[1:]):
        x = proc.ode_step(oracle(x, t), x, t, t_next)
    euler = (x - target).abs().max()
    assert heun < euler / 10


class CopyOracle(torch.nn.Module):
    """Two positions, tokens {0, 1}, joint mass 1/2 on (0, 0) and (1, 1). Exact posterior over clean tokens."""

    def __init__(self, mask_id=2):
        super().__init__()
        self.mask_id = mask_id
        self.unused = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, t, cond=None, drop_cond=None):
        p = torch.full((*x.shape, 3), 0.5, dtype=self.unused.dtype)
        p[..., 2] = 0
        for i in range(2):
            known = x[:, 1 - i] != self.mask_id
            p[known, i] = torch.nn.functional.one_hot(x[known, 1 - i], 3).to(p.dtype)
        return p.clamp(min=1e-30).log()


def test_masked_reference_samples_the_joint():
    proc = MaskedDiffusion(mask_id=2)
    g = torch.Generator().manual_seed(0)
    x = reference_sample(CopyOracle(), proc, (4000, 2), num_steps=200, generator=g)
    one_step = sample(CopyOracle(), proc, (4000, 2), num_steps=1, generator=g)
    assert (x != proc.mask_id).all()
    assert (x[:, 0] != x[:, 1]).double().mean() < 0.02    # high NFE recovers the coupling
    assert (one_step[:, 0] != one_step[:, 1]).double().mean() > 0.4  # one step factorises it


def test_reference_does_not_change_a_coupled_process():
    proc = MaskedDiffusion(mask_id=2, sampler="coupled", vocab_size=3)
    reference_sample(CopyOracle(), proc, (4, 2), num_steps=4)
    assert proc.sampler == "coupled"


def test_parity():
    x = torch.randn(4, 3)
    assert parity(x, x.clone()).passed and parity(x, x).max_abs == 0
    off = parity(x + 0.1, x, atol=0.05)
    assert not off.passed and off.max_abs == pytest.approx(0.1, abs=1e-6)
    tok = torch.tensor([[1, 2], [3, 4]])
    assert parity(tok, torch.tensor([[1, 2], [3, 5]]), min_match=0.75).passed
    assert not parity(tok, torch.tensor([[1, 2], [3, 5]])).passed
    with pytest.raises(ValueError, match="shape mismatch"):
        parity(x, x[:2])


def test_pipeline_reference_is_float64_and_ode_converges_to_it(tmp_path):
    cfg = tiny_config("configs/image_flow_grpo.yaml", tmp_path, "process.sampler=ode")
    pipe = Pipeline.from_config(cfg)
    torch.manual_seed(0)
    with torch.no_grad():  # a fresh model is zero-initialised at the output, so every sampler would agree
        for p in pipe.c.model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    ref = pipe.reference(n=3, num_steps=64, seed=1)
    assert ref.raw.dtype == torch.float64
    assert next(pipe.c.model.parameters()).dtype == torch.float32  # the pipeline's model is untouched
    fine = pipe.generate(n=3, num_steps=128, seed=1).raw
    coarse = pipe.generate(n=3, num_steps=2, seed=1).raw
    assert parity(fine, ref.raw.float()).rmse < parity(coarse, ref.raw.float()).rmse
