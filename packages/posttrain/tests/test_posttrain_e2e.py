"""Every shipped config runs SFT -> RL -> inference end to end at toy size."""

import numpy as np
import torch
from diffusion_omni.core import Pipeline, step
from diffusion_omni.core.testing import tiny_config
from diffusion_omni.posttrain import RLTrainer, SFTTrainer
from diffusion_omni.posttrain.rl.algorithms import compute_advantages


def test_sft_loss_is_finite(config_path, tmp_path):
    trainer = SFTTrainer(tiny_config(config_path, tmp_path))
    assert all(torch.isfinite(torch.tensor(trainer.train_step())) for _ in range(2))


def test_rollout_log_probs_are_on_policy(config_path, tmp_path):
    # Rescoring a fresh rollout with the same weights must give ratio == 1, otherwise
    # the PPO objective is silently off-policy (CFG, time or conditioning mismatch).
    trainer = RLTrainer(tiny_config(config_path, tmp_path, "sampling.guidance_scale=2.0"))
    ro = trainer.rollout()
    c, traj = trainer.c, ro.traj
    for k in traj.stochastic.nonzero().flatten().tolist():
        tr = step(trainer.model, c.process, traj.xs[:, k], traj.timesteps[k], traj.timesteps[k + 1], ro.cond,
                  c.sampling.guidance_scale, x_next=traj.xs[:, k + 1])
        torch.testing.assert_close(tr.log_prob, traj.log_probs[:, k], atol=1e-4, rtol=1e-4)


def test_sft_rl_sample(config_path, tmp_path):
    cfg = tiny_config(config_path, tmp_path)
    sft_ckpt = SFTTrainer(cfg).train()

    cfg["rl"]["init_from"] = str(sft_ckpt)
    trainer = RLTrainer(cfg)
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    rl_ckpt = trainer.train()
    after = trainer.model.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before if k in after)

    pipe = Pipeline.from_checkpoint(rl_ckpt, device="cpu")
    gen = pipe.generate(n=3, num_steps=3, seed=1)
    assert len(gen.samples) == 3
    if isinstance(gen.samples, list):
        assert all(isinstance(s, str) for s in gen.samples)
    else:
        assert gen.samples.shape == (3, *pipe.c.sample_shape)
        assert gen.samples.min() >= 0 and gen.samples.max() <= 1
    # same seed, same output
    again = pipe.generate(prompts=gen.prompts, num_steps=3, seed=1)
    assert torch.equal(gen.raw, again.raw)


def test_plugin_components_from_config(configs, tmp_path):
    cfg = tiny_config(configs["video_flow_grpo"], tmp_path, "plugins=[examples.custom_plugin]", "rl.reward={type: centered_blob}")
    ro = RLTrainer(cfg).rollout()
    assert ro.rewards.shape == (4,) and ((ro.rewards >= 0) & (ro.rewards <= 1)).all()


def test_group_advantages():
    r = np.array([1.0, 3.0, 5.0, 5.0])
    adv = compute_advantages(r, np.array([0, 0, 1, 1]), "group")
    np.testing.assert_allclose(adv, [-1, 1, 0, 0], atol=1e-3)
    assert abs(compute_advantages(r, np.zeros(4), "batch").mean()) < 1e-9
