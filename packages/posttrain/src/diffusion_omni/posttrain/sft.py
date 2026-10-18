"""Supervised denoising training. Produces the base policy that RL starts from."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from diffusion_omni.core.build import Components, build
from diffusion_omni.core.config import from_dict
from diffusion_omni.core.registry import DATASETS
from diffusion_omni.core.utils import JsonlLogger, load_weights, save_checkpoint, seed_everything


@dataclass
class SFTConfig:
    dataset: dict = None
    init_from: str | None = None
    steps: int = 1000
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 100
    cond_dropout: float = 0.1  # label dropout so the model also learns the unconditional branch for CFG
    max_grad_norm: float = 1.0
    log_every: int = 50
    save_every: int = 0


class SFTTrainer:
    def __init__(self, cfg: dict, components: Components | None = None):
        self.cfg = cfg
        self.c = components or build(cfg)
        self.sft = from_dict(SFTConfig, cfg.get("sft"))
        if self.sft.dataset is None:
            raise ValueError("sft.dataset is required")
        seed = cfg.get("seed", 0)
        seed_everything(seed)
        self.rng = np.random.default_rng(seed)
        self.out_dir = Path(cfg.get("output_dir", "runs/default")) / "sft"
        self.dataset = DATASETS.build(self.sft.dataset, **self.c.context)
        if self.sft.init_from:
            self.c.model.load_state_dict(load_weights(self.sft.init_from, self.c.device))
        self.opt = torch.optim.AdamW(self.c.model.parameters(), lr=self.sft.lr, weight_decay=self.sft.weight_decay)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: min(1.0, (s + 1) / max(1, self.sft.warmup_steps)))

    def train_step(self) -> float:
        c, sft = self.c, self.sft
        x0, cond = self.dataset.sample(sft.batch_size, self.rng)
        x0 = x0.to(c.device)
        cond = {k: v.to(c.device) for k, v in cond.items()}
        drop = None
        if c.model.supports_guidance and sft.cond_dropout > 0:
            drop = torch.rand(x0.shape[0], device=c.device) < sft.cond_dropout
        loss = c.process.training_loss(c.model, x0, cond, drop)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(c.model.parameters(), sft.max_grad_norm)
        self.opt.step()
        self.sched.step()
        return loss.item()

    def train(self) -> Path:
        logger = JsonlLogger(self.out_dir / "log.jsonl")
        self.c.model.train()
        running = []
        for s in range(self.sft.steps):
            running.append(self.train_step())
            if (s + 1) % self.sft.log_every == 0 or s + 1 == self.sft.steps:
                logger.log(s + 1, {"loss": float(np.mean(running)), "lr": self.sched.get_last_lr()[0]})
                running = []
            if self.sft.save_every and (s + 1) % self.sft.save_every == 0:
                save_checkpoint(self.out_dir / f"step_{s + 1}", self.c.model.state_dict(), self.cfg)
        return save_checkpoint(self.out_dir / "final", self.c.model.state_dict(), self.cfg)
