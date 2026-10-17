import copy
from pathlib import Path

import pytest
import torch

from omnidiff.core import Pipeline, sample
from omnidiff.core.config import load_config
from omnidiff.core.testing import tiny_config
from omnidiff.distill import DistillTrainer
from omnidiff.distill.trainer import EMA
from omnidiff.posttrain import SFTTrainer

ROOT = Path(__file__).resolve().parents[3]
DISTILL_CONFIGS = sorted((ROOT / "configs" / "distill").glob("*.yaml"))
TINY_DISTILL = ["distill.steps=2", "distill.batch_size=4", "distill.pool_size=8", "distill.teacher_steps=4",
                "distill.log_every=1", "distill.eval_every=2", "distill.eval_n=4"]


@pytest.fixture(scope="module")
def teachers(tmp_path_factory):
    """Tiny SFT teachers, keyed by the base config stem the distill configs point at."""
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


@pytest.mark.parametrize("path", DISTILL_CONFIGS, ids=lambda p: p.stem)
def test_distill_config_end_to_end(path, teachers, tmp_path):
    cfg = distill_config(path, teachers, tmp_path)
    tr = DistillTrainer(cfg)
    before = {k: v.clone() for k, v in tr.student.state_dict().items()}
    ckpt = tr.train()
    assert any(not torch.equal(before[k], v) for k, v in tr.student.state_dict().items())
    pipe = Pipeline.from_checkpoint(ckpt, device="cpu")
    assert pipe.c.sampling.timesteps == [round(float(t), 6) for t in tr.grid]
    gen = pipe.generate(n=3, seed=0)
    assert len(gen.samples) == 3
    if isinstance(gen.samples, list):
        assert all(isinstance(s, str) for s in gen.samples)
    else:
        assert gen.samples.shape == (3, *pipe.c.sample_shape)


def _trainer(teachers, tmp_path, name, *extra):
    return DistillTrainer(distill_config(ROOT / "configs" / "distill" / f"{name}.yaml", teachers, tmp_path, *extra))


def test_dmd2_generator_gradient_is_the_score_difference(teachers, tmp_path):
    tr = _trainer(teachers, tmp_path, "image_flow_dmd2")
    m = tr.method
    torch.manual_seed(0)
    with torch.no_grad():  # make the fake score differ from the teacher
        for p in m.fake["score"].parameters():
            p.add_(0.02 * torch.randn_like(p))
    tau, eps = torch.full((4,), 0.5), torch.randn(4, *tr.shape)
    x0, cond = torch.randn(4, *tr.shape, requires_grad=True), tr.prompt_cond(4)
    m._noisy = lambda tr_, x: (tr_.process.q_sample(x, tau, eps), tau)
    m.generate = lambda tr_, n: (x0, cond)
    loss, _ = m.loss("student", tr)
    loss.backward()
    with torch.no_grad():
        x_tau = tr.process.q_sample(x0, tau, eps)
        real = tr.process.to_x0(tr.teacher_pred(x_tau, tau, cond), x_tau, tau)
        fake = tr.process.to_x0(m.fake["score"](x_tau, tau, cond), x_tau, tau)
        w = 1.0 / ((x0 - real).abs().flatten(1).mean(1) + 1e-6)
        expected = (fake - real) * w.view(-1, 1, 1, 1) / x0.numel()
    torch.testing.assert_close(x0.grad, expected)


def test_flow_map_student_starts_as_teacher_euler(teachers, tmp_path):
    tr = _trainer(teachers, tmp_path, "image_flow_flow_map")
    cond = tr.prompt_cond(4)
    kw = dict(cond=cond, timesteps=tr.grid, generator=None)
    torch.manual_seed(0)
    student = sample(tr.student, tr.student_process, (4, *tr.shape), **kw)
    torch.manual_seed(0)
    teacher = sample(tr.teacher, tr.teacher_process, (4, *tr.shape), **kw)
    torch.testing.assert_close(student, teacher, atol=1e-5, rtol=1e-5)


def test_flow_map_jvp_matches_finite_difference(teachers, tmp_path):
    tr = _trainer(teachers, tmp_path, "image_flow_flow_map")
    with torch.no_grad():  # move s_embed off zero so s actually matters
        for p in tr.student.s_embed.parameters():
            p.add_(0.05 * torch.randn_like(p))
    b = tr.data.batch(4)
    t, s = torch.tensor([0.9, 0.7, 0.5, 0.3]), torch.tensor([0.2, 0.0, 0.4, 0.1])
    z = tr.process.q_sample(b.x0, t, torch.randn_like(b.x0))
    v = tr.teacher_pred(z, t, b.cond)
    m = tr.method
    m.derivative = "jvp"
    _, d_jvp = m.average_velocity(tr, z, t, s, v, b.cond)
    m.derivative, m.fd_eps = "finite_difference", 1e-3
    _, d_fd = m.average_velocity(tr, z, t, s, v, b.cond)
    torch.testing.assert_close(d_jvp, d_fd, atol=5e-3, rtol=5e-2)


def test_flow_map_finite_difference_stays_inside_the_time_range(teachers, tmp_path):
    tr = _trainer(teachers, tmp_path, "image_flow_flow_map", "distill.method.derivative=finite_difference",
                  "distill.method.fd_eps=0.05")
    b = tr.data.batch(2)
    t, s = torch.tensor([0.02, 0.6]), torch.tensor([0.0, 0.3])
    z = tr.process.q_sample(b.x0, t, torch.randn_like(b.x0))
    v = tr.teacher_pred(z, t, b.cond)
    seen = []
    student = tr.student
    tr.student = lambda z_, t_, cond, s=None: (seen.append(t_.clone()), student(z_, t_, cond, s=s))[1]
    _, dudt = tr.method.average_velocity(tr, z, t, s, v, b.cond)
    tr.student = student
    assert torch.isfinite(dudt).all()
    torch.testing.assert_close(seen[1], torch.tensor([0.07, 0.55]))  # forward where t < eps, else backward
    with torch.no_grad():
        fwd = (student(z[:1] + 0.05 * v[:1], t[:1] + 0.05, {k: c[:1] for k, c in b.cond.items()}, s=s[:1])
               - student(z[:1], t[:1], {k: c[:1] for k, c in b.cond.items()}, s=s[:1])) / 0.05
    torch.testing.assert_close(dudt[:1], fwd, atol=1e-5, rtol=1e-4)


def test_sdtt_targets_are_distributions_and_rounds_advance(teachers, tmp_path):
    tr = _trainer(teachers, tmp_path, "text_masked_sdtt", "distill.steps=3")
    m = tr.method
    b = tr.data.batch(4)
    x_t = tr.process.q_sample(b.x0, 0.8)
    target = m.targets(tr, x_t, torch.tensor(0.8), torch.tensor(0.6), b.cond)
    masked = x_t == tr.process.mask_id
    torch.testing.assert_close(target[masked].sum(-1), torch.ones(int(masked.sum())))
    assert (target[..., tr.process.mask_id] == 0).all()
    with torch.no_grad():
        for p in tr.student.parameters():
            p.add_(0.01)
    m.schedule(1, tr)  # round_len = 1, so step 1 starts round 1 and snapshots the student
    assert m.round == 1
    ema = tr.roles["student"].ema
    with ema.swapped_in(tr.student):
        expected = {k: v.clone() for k, v in tr.student.state_dict().items()}
    for k, v in m.round_teacher.state_dict().items():
        torch.testing.assert_close(v, expected[k])


def test_consistency_boundary_target_is_the_teacher_step(teachers, tmp_path):
    tr = _trainer(teachers, tmp_path, "image_flow_consistency")
    x = torch.randn(2, *tr.shape)
    pred = tr.student_pred(x, torch.zeros(2), tr.prompt_cond(2))
    torch.testing.assert_close(tr.process.to_x0(pred, x, torch.zeros(2)), x)


@pytest.mark.parametrize("name", ["image_flow_kd", "image_flow_dmd2", "text_masked_sdtt"])
def test_distill_save_resume_is_bit_identical(name, teachers, tmp_path):
    extra = ["distill.steps=4", "distill.log_every=100", "distill.eval_every=0"]
    if name == "image_flow_dmd2":
        extra.append("distill.method.fake_updates=1")

    full = _trainer(teachers, tmp_path / "full", name, *extra)
    full.train()
    full_student = {k: v.clone() for k, v in full.student.state_dict().items()}
    full_ema = None if full.roles["student"].ema is None else {
        k: v.clone() for k, v in full.roles["student"].ema.shadow.items()
    }

    split = _trainer(teachers, tmp_path / "split", name, *extra[:])
    split.dc.steps = 2
    split.train()
    state = split.save_state(split.out_dir / "state")
    resumed_cfg = distill_config(ROOT / "configs" / "distill" / f"{name}.yaml", teachers, tmp_path / "split",
                                  *extra, f"distill.resume_from={state}")
    resumed = DistillTrainer(resumed_cfg)
    resumed.train()

    for k, v in full_student.items():
        torch.testing.assert_close(resumed.student.state_dict()[k], v, atol=0, rtol=0)
    if full_ema is not None:
        for k, v in full_ema.items():
            torch.testing.assert_close(resumed.roles["student"].ema.shadow[k], v, atol=0, rtol=0)


def test_ema_swap_restores_weights():
    model = torch.nn.Linear(3, 3)
    ema = EMA(model, 0.5)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    live = model.weight.clone()
    with ema.swapped_in(model):
        torch.testing.assert_close(model.weight, ema.shadow["weight"])
    torch.testing.assert_close(model.weight, live)


def test_renoise_student_is_rl_tunable(teachers, tmp_path):
    """A few-step student exported with the renoise sampler is a policy with exact log-probs:
    posttrain's RL trainer runs on it unchanged, and a fresh rollout is on-policy."""
    from omnidiff.core.sampling import step
    from omnidiff.posttrain import RLTrainer

    ckpt = _trainer(teachers, tmp_path, "image_flow_dmd2").train()
    cfg = load_config(ckpt / "config.yaml", ["device=cpu", f"output_dir={tmp_path}/rl", f"rl.init_from={ckpt}",
                                             "rl.epochs=1", "rl.prompts_per_epoch=2", "rl.group_size=2",
                                             "rl.rollout_batch_size=4", "rl.train_batch_size=2"])
    tr = RLTrainer(cfg)
    ro = tr.rollout()
    assert ro.traj.xs.shape[1] == len(cfg["sampling"]["timesteps"])
    assert ro.traj.stochastic.tolist() == [True] * (len(cfg["sampling"]["timesteps"]) - 2) + [False]
    c, traj = tr.c, ro.traj
    for k in range(traj.xs.shape[1] - 2):
        tr_k = step(tr.model, c.process, traj.xs[:, k], traj.timesteps[k], traj.timesteps[k + 1], ro.cond,
                    x_next=traj.xs[:, k + 1])
        torch.testing.assert_close(tr_k.log_prob, traj.log_probs[:, k], atol=1e-4, rtol=1e-4)
    tr.update(ro)


@pytest.mark.parametrize("noise", ["gumbel", "uniform"])
def test_coupled_kd_student_starts_as_the_coupled_teacher(noise, teachers, tmp_path):
    """At init the noise projection is zero, so with coupled_gumbel = 1 the student is the teacher's
    coupled sampler; the pool's samples are reproduced exactly when student and teacher share the grid."""
    tr = _trainer(teachers, tmp_path, "text_masked_coupled_kd", "distill.student_steps=4", "distill.teacher_steps=4",
                  f"distill.method.noise={noise}")
    assert tr.teacher_process.coupled_noise == tr.student_process.coupled_noise == noise
    assert tr.data.cond["token_noise"].dim() == (3 if noise == "gumbel" else 2)
    b = tr.data.batch(4)
    proc = copy.copy(tr.student_process)
    proc.coupled_gumbel = 1.0
    out = sample(tr.student, proc, b.x0.shape, cond=b.cond, timesteps=tr.grid)
    torch.testing.assert_close(out, b.x0)
