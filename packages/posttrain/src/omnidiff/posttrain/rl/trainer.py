"""Online RL post-training: roll out -> score -> advantages -> clipped policy-gradient updates."""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from omnidiff.core.build import Components, build
from omnidiff.core.config import from_dict
from omnidiff.core.nn.lora import apply_lora, has_lora, lora_disabled, merged_state_dict
from omnidiff.core.registry import ALGORITHMS, REWARDS
from omnidiff.core.sampling import Trajectory, sample, step
from omnidiff.core.utils import JsonlLogger, index_cond, load_weights, make_generator, save_checkpoint, seed_everything


@dataclass
class RLConfig:
    reward: dict = None
    algorithm: dict = field(default_factory=lambda: {"type": "grpo"})
    init_from: str | None = None
    lora: dict | None = None             # {rank, alpha, targets}; null = full fine-tune
    epochs: int = 100
    prompts_per_epoch: int = 8
    group_size: int = 4                  # samples per prompt
    rollout_batch_size: int = 32
    train_batch_size: int = 16
    inner_epochs: int = 1
    timestep_fraction: float = 1.0       # fraction of stochastic steps trained per minibatch
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    save_every: int = 0


@dataclass
class Rollout:
    traj: Trajectory
    cond: dict
    prompts: list
    rewards: np.ndarray
    advantages: torch.Tensor


class RLTrainer:
    def __init__(self, cfg: dict, components: Components | None = None):
        self.cfg = cfg
        self.c = components or build(cfg)
        self.rl = from_dict(RLConfig, cfg.get("rl"))
        if self.rl.reward is None:
            raise ValueError("rl.reward is required")
        seed = cfg.get("seed", 0)
        seed_everything(seed)
        self.out_dir = Path(cfg.get("output_dir", "runs/default")) / "rl"

        model = self.c.model
        if self.rl.init_from:
            model.load_state_dict(load_weights(self.rl.init_from, self.c.device))
        if self.rl.lora:
            apply_lora(model, **self.rl.lora)
        self.algo = ALGORITHMS.build(self.rl.algorithm)
        self.ref_model = None
        if self.algo.kl_coef > 0 and not has_lora(model):
            self.ref_model = copy.deepcopy(model).requires_grad_(False)
        self.reward = REWARDS.build(self.rl.reward, **self.c.context)
        params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=self.rl.lr, weight_decay=self.rl.weight_decay)
        self.rng = np.random.default_rng(seed)
        self.gen = make_generator(seed, self.c.device)
        self.cpu_gen = make_generator(seed, "cpu")

    @property
    def model(self):
        return self.c.model

    @contextlib.contextmanager
    def _reference(self):
        if self.ref_model is not None:
            yield self.ref_model
        else:
            with lora_disabled(self.model) as m:
                yield m

    def rollout(self) -> Rollout:
        c, rl = self.c, self.rl
        prompts = [p for p in c.prompts.sample(rl.prompts_per_epoch, self.rng) for _ in range(rl.group_size)]
        groups = np.repeat(np.arange(rl.prompts_per_epoch), rl.group_size)
        cond = c.prompts.encode(prompts, c.device)
        self.model.eval()
        trajs = []
        for i in range(0, len(prompts), rl.rollout_batch_size):
            sub = index_cond(cond, slice(i, i + rl.rollout_batch_size))
            n = len(prompts[i: i + rl.rollout_batch_size])
            trajs.append(sample(self.model, c.process, (n, *c.sample_shape), sub, c.sampling.num_steps,
                                c.sampling.guidance_scale, generator=self.gen, keep_trajectory=True))
        traj = Trajectory.cat(trajs)
        if not traj.stochastic.any():
            raise ValueError("sampler is deterministic (eta=0 / noise_level=0); RL needs a stochastic policy")
        rewards = np.asarray(self.reward(c.codec.decode(traj.samples), prompts), dtype=np.float64)
        adv = torch.as_tensor(self.algo.advantages(rewards, groups), dtype=torch.float32, device=c.device)
        return Rollout(traj, cond, prompts, rewards, adv)

    def update(self, ro: Rollout) -> dict[str, float]:
        c, rl, traj = self.c, self.rl, ro.traj
        self.model.train()
        steps = traj.stochastic.nonzero().flatten().tolist()
        n_steps = max(1, round(len(steps) * rl.timestep_fraction))
        ts, gs = traj.timesteps, c.sampling.guidance_scale
        totals: dict[str, float] = {}
        count = 0
        for _ in range(rl.inner_epochs):
            perm = torch.randperm(traj.xs.shape[0], generator=self.cpu_gen).to(c.device)
            for mb in perm.split(rl.train_batch_size):
                cond = index_cond(ro.cond, mb)
                self.opt.zero_grad(set_to_none=True)
                for k in sorted(self.rng.choice(steps, n_steps, replace=False).tolist()):
                    x_t, x_next = traj.xs[mb, k], traj.xs[mb, k + 1]
                    tr = step(self.model, c.process, x_t, ts[k], ts[k + 1], cond, gs, x_next=x_next)
                    kl = None
                    if self.algo.kl_coef > 0:
                        with torch.no_grad(), self._reference() as ref:
                            ref_tr = step(ref, c.process, x_t, ts[k], ts[k + 1], cond, gs, x_next=x_next)
                        kl = c.process.kl(tr, ref_tr)
                    out = self.algo.loss(tr.log_prob, traj.log_probs[mb, k], ro.advantages[mb], kl)
                    (out.loss / n_steps).backward()
                    for key, v in out.metrics.items():
                        totals[key] = totals.get(key, 0.0) + v
                    count += 1
                params = [p for p in self.model.parameters() if p.requires_grad]
                totals["grad_norm"] = totals.get("grad_norm", 0.0) + float(
                    torch.nn.utils.clip_grad_norm_(params, rl.max_grad_norm)) * n_steps
                self.opt.step()
        return {k: v / count for k, v in totals.items()}

    def save(self, name: str) -> Path:
        return save_checkpoint(self.out_dir / name, merged_state_dict(self.model), self.cfg)

    def train(self) -> Path:
        logger = JsonlLogger(self.out_dir / "log.jsonl")
        for epoch in range(self.rl.epochs):
            ro = self.rollout()
            metrics = self.update(ro)
            logger.log(epoch, {"reward": ro.rewards.mean(), "reward_std": ro.rewards.std(), **metrics})
            if self.rl.save_every and (epoch + 1) % self.rl.save_every == 0:
                self.save(f"epoch_{epoch + 1}")
        return self.save("final")
