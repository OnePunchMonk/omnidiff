"""Multi-role distillation trainer.

A distillation run has a frozen teacher, a student initialised from it, and,
depending on the method, auxiliary networks with their own optimisers (DMD2's
fake score and discriminator head). Each step the method names the roles to update
and in what order (two time-scale updates); the trainer freezes every other role
for that update, steps the optimiser, and keeps an EMA of the student.

The export is an ordinary omnidiff checkpoint whose config carries the student's
sampler and timestep grid, so ``Pipeline``, ``omnidiff sample`` and the scorecard
use it unchanged.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from omnidiff.core.build import build
from omnidiff.core.config import from_dict, load_config
from omnidiff.core.nn.denoiser import build_model
from omnidiff.core.nn.lora import apply_lora, merged_state_dict
from omnidiff.core.processes.base import ContinuousProcess
from omnidiff.core.registry import DATASETS, DISTILLERS, load_plugins
from omnidiff.core.sampling import predict, sample, step
from omnidiff.core.utils import (CONFIG, JsonlLogger, index_cond, load_weights, make_generator, resolve_device,
                                 save_checkpoint, seed_everything)


@dataclass
class DistillConfig:
    method: dict = None
    student_steps: int | None = None          # uniform student grid (process default spacing) ...
    student_timesteps: list[float] | None = None  # ... or an explicit one
    sampler: str | None = None                # student sampler; default comes from the method
    student_process: dict | None = None       # process settings for the student only, e.g. {clip_x0: 1.0}; exported
    teacher_guidance: float = 1.0             # CFG used for every teacher prediction (distilled into the student)
    data: dict | str = "teacher"              # "teacher": a pool of teacher ODE samples; or a dataset spec
    pool_size: int = 2048
    teacher_steps: int = 32
    steps: int = 2000                         # student updates
    batch_size: int = 32
    lr: float = 1e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    ema_decay: float | None = 0.99
    lora: dict | None = None
    log_every: int = 50
    eval_every: int = 0
    eval_n: int = 128
    save_every: int = 0


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].lerp_(v, 1 - self.decay)

    @contextlib.contextmanager
    def swapped_in(self, model: nn.Module):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict(self.shadow, strict=False)
        try:
            yield model
        finally:
            model.load_state_dict(backup, strict=False)


@dataclass
class Role:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    params: list[nn.Parameter]
    ema: EMA | None = None


@dataclass
class Batch:
    x0: torch.Tensor
    cond: dict
    states: torch.Tensor | None = None  # (B, K + 1, ...) teacher ODE states on the student grid (pool only)


class TeacherPool:
    """Teacher samples, generated once with the teacher's deterministic sampler (data-free distillation).

    Teacher states are also recorded on the student's grid, which trajectory methods (KD) regress from.
    """

    def __init__(self, tr: "DistillTrainer", size: int):
        grid = tr.grid.cpu().tolist()
        full = tr.process.timesteps(tr.dc.teacher_steps, "cpu").tolist()
        merged = sorted(set(round(t, 6) for t in full + grid), reverse=True)
        idx = [merged.index(round(t, 6)) for t in grid]
        prompts = tr.c.prompts.sample(size, tr.rng)
        cond = tr.c.prompts.encode(prompts, tr.device)
        xs = []
        for i in range(0, size, 256):
            sub = index_cond(cond, slice(i, i + 256))
            n = len(prompts[i: i + 256])
            traj = sample(tr.teacher, tr.teacher_process, (n, *tr.shape), sub, guidance_scale=tr.dc.teacher_guidance,
                          generator=tr.gen, keep_trajectory=True, timesteps=merged)
            xs.append(traj.xs[:, idx])
        self.states = torch.cat(xs)
        self.x0, self.cond, self.prompts = self.states[:, -1], cond, prompts
        self.rng = tr.rng

    def batch(self, n: int) -> Batch:
        i = torch.as_tensor(self.rng.integers(0, len(self.x0), n), device=self.x0.device)
        return Batch(self.x0[i], index_cond(self.cond, i), self.states[i])


class DatasetSource:
    def __init__(self, tr: "DistillTrainer", spec):
        self.dataset = DATASETS.build(spec, **tr.c.context)
        self.rng, self.device = tr.rng, tr.device

    def batch(self, n: int) -> Batch:
        x0, cond = self.dataset.sample(n, self.rng)
        return Batch(x0.to(self.device), {k: v.to(self.device) for k, v in cond.items()})


class DistillTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dc = from_dict(DistillConfig, cfg.get("distill"))
        if self.dc.method is None:
            raise ValueError("distill.method is required")
        load_plugins(cfg.get("plugins", []))
        seed = cfg.get("seed", 0)
        seed_everything(seed)
        self.rng = np.random.default_rng(seed)
        self.device = resolve_device(cfg.get("device", "auto"))
        self.gen = make_generator(seed, self.device)
        self.out_dir = Path(cfg.get("output_dir", "runs/distill")) / "distill"

        teacher_path = Path(cfg["teacher"])
        self.teacher_cfg = load_config(teacher_path / CONFIG)
        self.c = build(self.teacher_cfg, self.device)
        self.process, self.shape = self.c.process, self.c.sample_shape
        self.continuous = isinstance(self.process, ContinuousProcess)
        self.teacher = self.c.model
        self.teacher.load_state_dict(load_weights(teacher_path, self.device))
        self.teacher.eval().requires_grad_(False)

        self.method = DISTILLERS.build(self.dc.method)
        ptype = self.teacher_cfg["process"]["type"] if isinstance(self.teacher_cfg["process"], dict) else self.teacher_cfg["process"]
        if ptype not in self.method.processes:
            raise ValueError(f"{type(self.method).__name__} supports processes {self.method.processes}, not {ptype!r}")
        self.teacher_process = self._with_sampler("ode" if self.continuous else "ancestral")
        self.student_sampler = self.dc.sampler or (self.method.sampler if self.continuous else "ancestral")
        self.student_process = self._with_sampler(self.student_sampler)
        for k, v in (self.dc.student_process or {}).items():
            if not hasattr(self.student_process, k):
                raise KeyError(f"student_process: {type(self.process).__name__} has no setting {k!r}")
            setattr(self.student_process, k, v)
        self.grid = self._student_grid()

        student_spec = {**self.teacher_cfg["model"], "two_time": self.method.two_time}
        self.student = build_model(student_spec, **self.c.context).to(self.device)
        missing, unexpected = self.student.load_state_dict(self.teacher.state_dict(), strict=False)
        if unexpected or any(not k.startswith("s_embed.") for k in missing):
            raise RuntimeError(f"student/teacher mismatch: missing={missing} unexpected={unexpected}")
        if self.dc.lora:
            apply_lora(self.student, **self.dc.lora)
            if self.method.two_time:
                self.student.s_embed.requires_grad_(True)
        self.student.train()
        self.roles: dict[str, Role] = {}
        self.add_role("student", self.student, ema_decay=self.dc.ema_decay)

        self.data = TeacherPool(self, self.dc.pool_size) if self.dc.data == "teacher" else DatasetSource(self, self.dc.data)
        self.method.setup(self)

    # ---- construction helpers -------------------------------------------------------------------
    def _with_sampler(self, sampler: str):
        proc = copy.copy(self.process)
        proc.sampler = sampler
        return proc

    def _student_grid(self) -> torch.Tensor:
        if self.dc.student_timesteps is not None:
            ts = torch.tensor(self.dc.student_timesteps, dtype=torch.float32, device=self.device)
        else:
            ts = self.method.default_grid(self) if self.dc.student_steps is None else self.process.timesteps(
                self.dc.student_steps, self.device)
        return ts.to(self.device)

    def add_role(self, name: str, model: nn.Module, lr: float | None = None, betas=None, ema_decay: float | None = None) -> Role:
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr or self.dc.lr, betas=tuple(betas or self.dc.betas),
                                weight_decay=self.dc.weight_decay)
        role = Role(model, opt, params, EMA(model, ema_decay) if ema_decay else None)
        self.roles[name] = role
        return role

    # ---- shared operations for methods ------------------------------------------------------------
    @torch.no_grad()
    def teacher_pred(self, x, t, cond):
        return predict(self.teacher, x, t, cond, self.dc.teacher_guidance, self.process)

    def student_pred(self, x, t, cond, s=None, model=None):
        return predict(model or self.student, x, t, cond, 1.0, self.process, s=s)

    def prompt_cond(self, n: int) -> dict:
        return self.c.prompts.encode(self.c.prompts.sample(n, self.rng), self.device)

    @torch.no_grad()
    def student_rollout(self, cond: dict, k: int, n: int | None = None):
        """Run the student sampler from the prior for ``k`` steps; returns the state at ``grid[k]``."""
        n = n or next(iter(cond.values())).shape[0]
        x = self.process.prior((n, *self.shape), cond, self.device, self.gen)
        for j in range(k):
            x = step(self.student, self.student_process, x, self.grid[j], self.grid[j + 1], cond, generator=self.gen).x_next
        return x

    # ---- training ---------------------------------------------------------------------------------
    def _update(self, name: str) -> dict:
        role = self.roles[name]
        for other, r in self.roles.items():
            for p in r.params:
                p.requires_grad_(other == name)
        loss, metrics = self.method.loss(name, self)
        role.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        metrics[f"{name}_grad_norm"] = float(torch.nn.utils.clip_grad_norm_(role.params, self.dc.max_grad_norm))
        role.optimizer.step()
        if role.ema is not None:
            role.ema.update(role.model)
        self.method.after_update(name, self)
        metrics[f"{name}_loss"] = loss.item()
        return metrics

    def train(self) -> Path:
        logger = JsonlLogger(self.out_dir / "log.jsonl")
        totals: dict[str, list[float]] = {}
        for it in range(self.dc.steps):
            for name in self.method.schedule(it, self):
                for k, v in self._update(name).items():
                    totals.setdefault(k, []).append(v)
            last = it + 1 == self.dc.steps
            if (it + 1) % self.dc.log_every == 0 or last:
                row = {k: float(np.mean(v)) for k, v in totals.items()}
                if self.dc.eval_every and ((it + 1) % self.dc.eval_every == 0 or last):
                    row.update(self.evaluate())
                logger.log(it + 1, row)
                totals = {}
            if self.dc.save_every and (it + 1) % self.dc.save_every == 0:
                self.export(self.out_dir / f"step_{it + 1}")
        return self.export(self.out_dir / "final")

    @contextlib.contextmanager
    def eval_student(self):
        ema = self.roles["student"].ema
        with (ema.swapped_in(self.student) if ema else contextlib.nullcontext()):
            self.student.eval()
            try:
                yield self.student
            finally:
                self.student.train()

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Quick in-training check: student samples on the training grid, scored like the scorecard."""
        from omnidiff.reward.metrics import set_metrics

        rng = np.random.default_rng(0)
        prompts = self.c.prompts.sample(self.dc.eval_n, rng)
        cond = self.c.prompts.encode(prompts, self.device)
        with self.eval_student() as m:
            x = sample(m, self.student_process, (len(prompts), *self.shape), cond, timesteps=self.grid,
                       generator=make_generator(0, self.device))
        samples = self.c.codec.decode(x)
        reference = self.c.codec.decode(self.data.x0[: self.dc.eval_n]) if isinstance(self.data, TeacherPool) else None
        out = {f"eval_{k}": v for k, v in set_metrics(samples, prompts, reference).items()}
        sft = self.teacher_cfg.get("sft") or {}
        if sft.get("dataset"):
            dataset = DATASETS.build(sft["dataset"], **self.c.context)
            if hasattr(dataset, "oracle"):
                out.update({f"eval_{k}": v for k, v in dataset.oracle(samples, prompts).items()})
        return out

    def export(self, path) -> Path:
        cfg = copy.deepcopy(self.teacher_cfg)
        cfg["model"] = {**cfg["model"], "two_time": self.method.two_time}
        if isinstance(cfg["process"], str):
            cfg["process"] = {"type": cfg["process"]}
        cfg["process"]["sampler"] = self.student_sampler
        cfg["process"].update(self.dc.student_process or {})
        grid = [round(float(t), 6) for t in self.grid]
        cfg["sampling"] = {"num_steps": len(grid) - 1, "guidance_scale": 1.0, "timesteps": grid}
        cfg["distilled_from"] = {"teacher": str(self.cfg["teacher"]), "method": self.dc.method,
                                 "teacher_guidance": self.dc.teacher_guidance, "steps": self.dc.steps}
        with self.eval_student() as m:
            state = merged_state_dict(m)
        return save_checkpoint(path, state, cfg)
