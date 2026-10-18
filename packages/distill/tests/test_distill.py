import copy
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from diffusion_omni.core import Pipeline, sample
from diffusion_omni.core.config import load_config
from diffusion_omni.core.sampling import step
from diffusion_omni.core.testing import tiny_config
from diffusion_omni.distill import DistillTrainer
from diffusion_omni.distill.trainer import EMA
from diffusion_omni.posttrain import SFTTrainer

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
    cfg = distill_config(path, teachers, tmp_path, *(["distill.method.rounds=2"] if "progressive" in path.stem else []))
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


def test_dmd2_backward_simulation_states_lie_on_the_student_grid(teachers, tmp_path):
    """The generator is fed the student's own rollout state at grid[k], evaluated at t = grid[k], for every
    k < K (never the final grid point), so training sees the states the trained grid meets at inference."""
    from diffusion_omni.core.sampling import step

    tr = _trainer(teachers, tmp_path, "image_flow_dmd2", "distill.student_steps=4")
    m, calls = tr.method, []
    real_pred = tr.student_pred
    tr.student_pred = lambda x, t, cond, s=None, model=None: (calls.append((x, t)), real_pred(x, t, cond, s, model))[1]
    for stage in range(len(tr.grid) - 1):
        m._stage = lambda tr_, stage=stage: stage  # inject the stage instead of relying on random draws
        state = tr.gen.get_state()
        calls.clear()
        _, cond = m.generate(tr, 2)
        (x, t), = calls
        assert float(t) == float(tr.grid[stage])
        tr.gen.set_state(state)  # replay the rollout by hand with the same noise
        ref = tr.process.prior((2, *tr.shape), cond, tr.device, tr.gen)
        for j in range(stage):
            ref = step(tr.student, tr.student_process, ref, tr.grid[j], tr.grid[j + 1], cond, generator=tr.gen).x_next
        torch.testing.assert_close(x, ref)


def test_dmd2_rollout_equals_inference_sampling_and_is_detached(teachers, tmp_path):
    """student_rollout(k) is the inference sampler's state at grid[k], also on a non-uniform grid. It carries
    no gradient (as in DMD2, earlier steps are simulated without grad); only the final x0 prediction does."""
    tr = _trainer(teachers, tmp_path, "image_flow_dmd2", "distill.student_timesteps=[1.0,0.6,0.3,0.1,0.0]")
    assert tr.grid.tolist() == pytest.approx([1.0, 0.6, 0.3, 0.1, 0.0])
    cond = tr.prompt_cond(3)
    for k in range(len(tr.grid)):
        state = tr.gen.get_state()
        x = tr.student_rollout(cond, k, 3)
        tr.gen.set_state(state)
        traj = sample(tr.student, tr.student_process, (3, *tr.shape), cond, timesteps=tr.grid, generator=tr.gen,
                      keep_trajectory=True)
        torch.testing.assert_close(x, traj.xs[:, k])
        assert not x.requires_grad
    x0, _ = tr.method.generate(tr, 3)
    assert x0.requires_grad


def test_dmd2_last_stage_prob(teachers, tmp_path):
    tr = _trainer(teachers, tmp_path, "image_flow_dmd2")
    assert tr.method.last_stage_prob == 0.5
    K = len(tr.grid) - 1
    draws = [tr.method._stage(tr) for _ in range(4000)]
    assert set(draws) == set(range(K))
    assert draws.count(K - 1) / len(draws) == pytest.approx(0.5 + 0.5 / K, abs=0.03)  # p + (1 - p) / K
    tr.method.last_stage_prob = None
    draws = [tr.method._stage(tr) for _ in range(4000)]
    assert draws.count(K - 1) / len(draws) == pytest.approx(1 / K, abs=0.03)
    tr.method.last_stage_prob = 1.0
    assert {tr.method._stage(tr) for _ in range(20)} == {K - 1}


def test_dmd2_fake_and_generator_updates_share_the_stage_distribution(teachers, tmp_path):
    tr = _trainer(teachers, tmp_path, "image_flow_dmd2")
    m, stages = tr.method, []
    real = m._stage
    m._stage = lambda tr_: stages.append(real(tr_)) or stages[-1]
    m.loss("fake", tr)
    m.loss("student", tr)
    assert len(stages) == 2  # both roles go through generate -> _stage, one draw each


@pytest.mark.parametrize("p", [-0.1, 1.5, float("nan")])
def test_dmd2_last_stage_prob_is_validated(p):
    from diffusion_omni.distill.methods.dmd2 import DMD2

    with pytest.raises(ValueError):
        DMD2(last_stage_prob=p)
    DMD2(last_stage_prob=0.0), DMD2(last_stage_prob=1.0), DMD2()


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


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


def _rng_probe(tr):
    """Probe every RNG stream and restore it so the probe itself is non-mutating."""
    python_state = random.getstate()
    numpy_state = copy.deepcopy(np.random.get_state())
    torch_state = torch.get_rng_state()
    generator_state = tr.gen.get_state()
    trainer_state = copy.deepcopy(tr.rng.bit_generator.state)
    try:
        return {
            "python": [random.random() for _ in range(4)],
            "numpy": np.random.random(4),
            "trainer": tr.rng.integers(0, 100, 4),
            "torch": torch.rand(4),
            "generator": torch.rand(4, generator=tr.gen),
        }
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        tr.gen.set_state(generator_state)
        tr.rng.bit_generator.state = trainer_state


@pytest.mark.parametrize("name", ["image_flow_kd", "image_flow_dmd2", "image_flow_dmd2_reward", "text_masked_dcd",
                                   "text_masked_sdtt", "image_flow_consistency", "image_flow_rcm"])
def test_distill_save_resume_is_bit_identical(name, teachers, tmp_path):
    extra = ["distill.steps=4", "distill.log_every=100", "distill.eval_every=0"]
    if name == "image_flow_dmd2":
        extra.append("distill.method.fake_updates=1")
    if name == "image_flow_consistency":
        extra.append("distill.method.target=ema")
    if name == "image_flow_dmd2_reward":
        extra.extend(["distill.method.fake_updates=1", "distill.method.reward_weight=0.7"])

    full = _trainer(teachers, tmp_path / "full", name, *extra)
    full.train()
    full_roles = {name: {"model": copy.deepcopy(role.model.state_dict()),
                         "optimizer": copy.deepcopy(role.optimizer.state_dict()),
                         "ema": None if role.ema is None else copy.deepcopy(role.ema.shadow)}
                  for name, role in full.roles.items()}
    full_method = copy.deepcopy(full.method.state_dict())
    full_rng = _rng_probe(full)

    split = _trainer(teachers, tmp_path / "split", name, *extra[:])
    split.dc.steps = 2
    split.train()
    state = split.save_state(split.out_dir / "state")
    torch.load(state / "state.pt", map_location="cpu", weights_only=True)
    resumed_cfg = distill_config(ROOT / "configs" / "distill" / f"{name}.yaml", teachers, tmp_path / "split",
                                  *extra, f"distill.resume_from={state}")
    resumed = DistillTrainer(resumed_cfg)
    resumed.train()

    resumed_roles = {name: {"model": role.model.state_dict(),
                            "optimizer": role.optimizer.state_dict(),
                            "ema": None if role.ema is None else role.ema.shadow}
                     for name, role in resumed.roles.items()}
    _assert_nested_equal(full_roles, resumed_roles)
    _assert_nested_equal(full_method, resumed.method.state_dict())
    _assert_nested_equal(full_rng, _rng_probe(resumed))


def test_sdtt_mid_round_save_resume_is_bit_identical(teachers, tmp_path):
    """The SDTT round teacher must be restored after the second round has started."""
    extra = ["distill.steps=6", "distill.method.rounds=2", "distill.log_every=100", "distill.eval_every=0"]
    full = _trainer(teachers, tmp_path / "full", "text_masked_sdtt", *extra)
    full_ckpt = full.train()
    full_roles = {name: {"model": copy.deepcopy(role.model.state_dict()),
                         "optimizer": copy.deepcopy(role.optimizer.state_dict()),
                         "ema": None if role.ema is None else copy.deepcopy(role.ema.shadow)}
                  for name, role in full.roles.items()}
    full_method = copy.deepcopy(full.method.state_dict())

    split = _trainer(teachers, tmp_path / "split", "text_masked_sdtt", *extra)
    split.dc.steps = 4
    split.train()
    state = split.save_state(split.out_dir / "state")
    resumed_cfg = distill_config(ROOT / "configs" / "distill" / "text_masked_sdtt.yaml", teachers,
                                  tmp_path / "split", *extra, f"distill.resume_from={state}")
    resumed = DistillTrainer(resumed_cfg)
    resumed_ckpt = resumed.train()

    resumed_roles = {name: {"model": role.model.state_dict(),
                            "optimizer": role.optimizer.state_dict(),
                            "ema": None if role.ema is None else role.ema.shadow}
                     for name, role in resumed.roles.items()}
    _assert_nested_equal(full_roles, resumed_roles)
    _assert_nested_equal(full_method, resumed.method.state_dict())
    assert load_config(full_ckpt / "config.yaml")["sampling"]["timesteps"] == \
        load_config(resumed_ckpt / "config.yaml")["sampling"]["timesteps"]


def test_save_state_every_resume_uses_checkpoint_written_inside_train(teachers, tmp_path):
    extra = ["distill.steps=4", "distill.log_every=100", "distill.eval_every=0", "distill.save_state_every=2"]
    full = _trainer(teachers, tmp_path / "full", "image_flow_kd", *extra)
    full.train()
    split = _trainer(teachers, tmp_path / "split", "image_flow_kd", *extra)
    split.dc.steps = 2
    split.train()
    state = split.out_dir / "state"
    assert (state / "state.pt").exists()
    payload = torch.load(state / "state.pt", map_location="cpu", weights_only=True)
    assert payload["version"] == 1
    assert not (state / "state.pt.tmp").exists()

    resumed_cfg = distill_config(ROOT / "configs" / "distill" / "image_flow_kd.yaml", teachers,
                                  tmp_path / "split", *extra, f"distill.resume_from={state}")
    resumed = DistillTrainer(resumed_cfg)
    resumed.train()
    for name, role in full.roles.items():
        _assert_nested_equal(role.model.state_dict(), resumed.roles[name].model.state_dict())


def test_resume_builds_teacher_pool_from_saved_tensors(teachers, tmp_path):
    extra = ["distill.steps=3", "distill.log_every=100", "distill.eval_every=0"]
    split = _trainer(teachers, tmp_path / "split", "image_flow_kd", *extra)
    split.dc.steps = 1
    split.train()
    state = split.save_state(split.out_dir / "state")
    resumed_cfg = distill_config(ROOT / "configs" / "distill" / "image_flow_kd.yaml", teachers,
                                  tmp_path / "split", *extra, f"distill.resume_from={state}")
    with patch("diffusion_omni.distill.trainer.TeacherPool._generate", side_effect=AssertionError("pool regenerated")):
        resumed = DistillTrainer(resumed_cfg)
    assert torch.equal(resumed.data.states, split.data.states)


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
    from diffusion_omni.core.sampling import step
    from diffusion_omni.posttrain import RLTrainer

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


@pytest.mark.parametrize("guidance", [1.0, 2.0])
@pytest.mark.parametrize("name", ["image_flow_progressive", "image_ddpm_progressive"])
def test_progressive_target_is_two_teacher_steps(name, guidance, teachers, tmp_path):
    tr = _trainer(teachers, tmp_path, name, "distill.steps=3", f"distill.teacher_guidance={guidance}")
    m, p = tr.method, tr.process
    x_t, cond = torch.randn(3, *tr.shape), tr.prompt_cond(3)
    for t, mid, s in [(0.75, 0.625, 0.5), (0.25, 0.125, 0.0), (1.0, 0.75, 0.5), (1.0, 0.5, 0.0)]:  # data, and t = 1
        t, mid, s = torch.tensor(t), torch.tensor(mid), torch.tensor(s)
        x_s = m.two_teacher_steps(tr, x_t, t, mid, s, cond)
        # the same two steps through the library sampler
        x_mid = step(tr.teacher, tr.teacher_process, x_t, t, mid, cond, tr.dc.teacher_guidance).x_next
        torch.testing.assert_close(x_s, step(tr.teacher, tr.teacher_process, x_mid, mid, s, cond,
                                             tr.dc.teacher_guidance).x_next)
        pred = m.target_pred(p, x_t, x_s, t, s)
        torch.testing.assert_close(p.ode_step(pred, x_t, t, s), x_s, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(p.to_x0(pred, x_t, t), m.x0_eps_target(p, x_t, x_s, t, s)[0], atol=1e-4, rtol=1e-4)


def test_progressive_rounds_advance_and_halve_the_grid(teachers, tmp_path):
    tr = _trainer(teachers, tmp_path, "image_flow_progressive", "distill.steps=3")
    m = tr.method
    assert [len(m.grid(tr, r)) - 1 for r in range(3)] == [8, 4, 2]
    assert torch.equal(tr.grid, m.grid(tr, 0))
    assert m.round_teacher is None  # round 0 distils the real teacher
    for p, q in zip(tr.student.parameters(), tr.teacher.parameters()):  # the student starts as the teacher
        torch.testing.assert_close(p, q)
    tr._update("student")  # gives the optimiser state
    assert tr.roles["student"].optimizer.state
    with torch.no_grad():
        for p in tr.student.parameters():
            p.add_(0.01)
    tr.roles["student"].ema.update(tr.student)
    with tr.roles["student"].ema.swapped_in(tr.student):
        expected = {k: v.clone() for k, v in tr.student.state_dict().items()}
    m.schedule(1, tr)  # round_len = 1: step 1 starts round 1 and snapshots the EMA student
    assert m.round == 1 and torch.equal(tr.grid, m.grid(tr, 1))
    role = tr.roles["student"]
    for k, v in m.round_teacher.state_dict().items():  # snapshot = EMA; student and EMA restart from it
        torch.testing.assert_close(v, expected[k])
        torch.testing.assert_close(tr.student.state_dict()[k], v)
        if k in role.ema.shadow:
            torch.testing.assert_close(role.ema.shadow[k], v)
    assert not role.optimizer.state
    sg = m.grid(tr, 1)
    t, mid, s = m.sample_triple(tr, sg, tr.process.timesteps(2 * (len(sg) - 1), "cpu"), 64)
    assert ((mid < t) & (mid > s)).all()
    assert all(float(v) in set(tr.process.timesteps(8, "cpu").tolist()) for v in mid)
    loss, metrics = m.loss("student", tr)
    assert metrics["student_steps"] == 4 and torch.isfinite(loss)


def test_progressive_training_ends_on_the_last_rounds_grid(teachers, tmp_path):
    tr = _trainer(teachers, tmp_path, "image_flow_progressive", "distill.steps=3")
    tr.train()
    assert tr.method.round == tr.method.rounds - 1
    assert torch.equal(tr.grid, tr.method.grid(tr, tr.method.rounds - 1)) and len(tr.grid) == 3


@pytest.mark.parametrize("split", [3, 5], ids=["mid_round_1", "final_round"])
def test_progressive_save_resume_is_bit_identical(split, teachers, tmp_path):
    """A checkpoint preserves both progressive round state and the grid at either split point."""
    extra = ["distill.steps=6", "distill.method.rounds=3", "distill.log_every=100", "distill.eval_every=0"]
    full = _trainer(teachers, tmp_path / "full", "image_flow_progressive", *extra)
    full_ckpt = full.train()
    full_roles = {name: {"model": copy.deepcopy(role.model.state_dict()),
                         "optimizer": copy.deepcopy(role.optimizer.state_dict()),
                         "ema": None if role.ema is None else copy.deepcopy(role.ema.shadow)}
                  for name, role in full.roles.items()}
    full_method = copy.deepcopy(full.method.state_dict())
    full_grid = full.grid.clone()
    full_rng = _rng_probe(full)

    split_tr = _trainer(teachers, tmp_path / "split", "image_flow_progressive", *extra)
    split_tr.dc.steps = split
    split_tr.train()
    state = split_tr.save_state(split_tr.out_dir / "state")
    torch.load(state / "state.pt", map_location="cpu", weights_only=True)
    resumed_cfg = distill_config(ROOT / "configs" / "distill" / "image_flow_progressive.yaml", teachers,
                                  tmp_path / "split", *extra, f"distill.resume_from={state}")
    resumed = DistillTrainer(resumed_cfg)
    resumed_ckpt = resumed.train()

    resumed_roles = {name: {"model": role.model.state_dict(),
                            "optimizer": role.optimizer.state_dict(),
                            "ema": None if role.ema is None else role.ema.shadow}
                     for name, role in resumed.roles.items()}
    _assert_nested_equal(full_roles, resumed_roles)
    _assert_nested_equal(full_method, resumed.method.state_dict())
    _assert_nested_equal(full_grid, resumed.grid)
    _assert_nested_equal(full_rng, _rng_probe(resumed))
    assert load_config(full_ckpt / "config.yaml")["sampling"]["timesteps"] == \
        load_config(resumed_ckpt / "config.yaml")["sampling"]["timesteps"]


def test_progressive_validates_steps_and_grid_overrides(teachers, tmp_path):
    with pytest.raises(ValueError, match="steps >= rounds"):
        _trainer(teachers, tmp_path, "image_flow_progressive", "distill.steps=2")
    with pytest.raises(ValueError, match="student_steps"):
        _trainer(teachers, tmp_path, "image_flow_progressive", "distill.steps=3", "distill.student_steps=4")
