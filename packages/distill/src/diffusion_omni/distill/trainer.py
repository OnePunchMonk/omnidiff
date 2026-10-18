"""Multi-role distillation trainer.

A distillation run has a frozen teacher, a student initialised from it, and,
depending on the method, auxiliary networks with their own optimisers (DMD2's
fake score and discriminator head). Each step the method names the roles to update
and in what order (two time-scale updates); the trainer freezes every other role
for that update, steps the optimiser, and keeps an EMA of the student.

The export is an ordinary diffusion-omni checkpoint whose config carries the student's
sampler and timestep grid, so ``Pipeline``, ``diffusion-omni sample`` and the scorecard
use it unchanged.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusion_omni.core.build import build
from diffusion_omni.core.config import from_dict, load_config
from diffusion_omni.core.nn.denoiser import build_model
from diffusion_omni.core.nn.lora import apply_lora, merged_state_dict
from diffusion_omni.core.processes.base import ContinuousProcess
from diffusion_omni.core.registry import DATASETS, DISTILLERS, load_plugins
from diffusion_omni.core.sampling import predict, sample, step
from diffusion_omni.core.utils import (
    CONFIG,
    WEIGHTS,
    JsonlLogger,
    index_cond,
    load_weights,
    make_generator,
    resolve_device,
    save_checkpoint,
    seed_everything,
)

_STATE_MARKER = "__diffusion_omni_state_type__"


def _safe_encode(value):
    """Convert checkpoint state to values supported by ``weights_only=True``."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return {
            _STATE_MARKER: "numpy_array",
            "dtype": value.dtype.str,
            "value": torch.as_tensor(value).clone(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {_safe_encode(k): _safe_encode(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return {_STATE_MARKER: "tuple", "value": [_safe_encode(v) for v in value]}
    if isinstance(value, list):
        return [_safe_encode(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot safely checkpoint value of type {type(value).__name__}")


def _safe_decode(value):
    if isinstance(value, dict):
        marker = value.get(_STATE_MARKER)
        if marker == "numpy_array":
            return value["value"].numpy().astype(np.dtype(value["dtype"]), copy=False)
        if marker == "tuple":
            return tuple(_safe_decode(v) for v in value["value"])
        return {_safe_decode(k): _safe_decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_decode(v) for v in value]
    return value


def _canonical_json(value) -> str:
    """Serialize config with stable ordering and no YAML/Python-specific objects."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


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
    save_state_every: int = 0
    resume_from: str | None = None


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

    def __init__(self, tr: "DistillTrainer", size: int, saved_state=None):
        self.kind = "teacher"
        self.rng = tr.rng
        if saved_state is not None:
            self._load_state(saved_state, tr.device)
            return
        self._generate(tr, size)

    def _generate(self, tr: "DistillTrainer", size: int):
        grid = tr.grid.cpu().tolist()
        full = tr.process.timesteps(tr.dc.teacher_steps, "cpu").tolist()
        merged = sorted(set(round(t, 6) for t in full + grid), reverse=True)
        idx = [merged.index(round(t, 6)) for t in grid]
        prompts = tr.c.prompts.sample(size, tr.rng)
        cond = tr.c.prompts.encode(prompts, tr.device)
        if getattr(tr.teacher_process, "sampler", None) == "coupled":  # keep the noise: it defines each sample
            cond.update(tr.teacher_process.init_noise((size, *tr.shape), tr.device, tr.gen))
        xs = []
        for i in range(0, size, 256):
            sub = index_cond(cond, slice(i, i + 256))
            n = len(prompts[i: i + 256])
            traj = sample(tr.teacher, tr.teacher_process, (n, *tr.shape), sub, guidance_scale=tr.dc.teacher_guidance,
                          generator=tr.gen, keep_trajectory=True, timesteps=merged)
            xs.append(traj.xs[:, idx])
        self.states = torch.cat(xs)
        self.x0, self.cond, self.prompts = self.states[:, -1], cond, prompts

    def _load_state(self, state, device):
        saved = _safe_decode(state)
        self.states = saved["states"].to(device)
        self.x0 = saved["x0"].to(device)
        self.cond = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in saved["cond"].items()}
        self.prompts = saved["prompts"]

    def batch(self, n: int) -> Batch:
        i = torch.as_tensor(self.rng.integers(0, len(self.x0), n), device=self.x0.device)
        return Batch(self.x0[i], index_cond(self.cond, i), self.states[i])


class DatasetSource:
    def __init__(self, tr: "DistillTrainer", spec):
        self.kind = "dataset"
        self.dataset = DATASETS.build(spec, **tr.c.context)
        self.rng, self.device = tr.rng, tr.device

    def batch(self, n: int) -> Batch:
        x0, cond = self.dataset.sample(n, self.rng)
        return Batch(x0.to(self.device), {k: v.to(self.device) for k, v in cond.items()})

    def state_dict(self) -> dict:
        hook = getattr(self.dataset, "state_dict", None)
        return {} if hook is None else (hook() or {})

    def load_state_dict(self, state: dict) -> None:
        hook = getattr(self.dataset, "load_state_dict", None)
        if hook is None:
            if state:
                raise ValueError("dataset checkpoint contains state, but the dataset has no load_state_dict() hook")
            return
        hook(state)


class DistillTrainer:
    def __init__(self, cfg: dict, resume_from=None):
        self.cfg = cfg
        self.dc = from_dict(DistillConfig, cfg.get("distill"))
        if self.dc.method is None:
            raise ValueError("distill.method is required")
        load_plugins(cfg.get("plugins", []))
        seed = cfg.get("seed", 0)
        seed_everything(seed)
        self.rng = np.random.default_rng(seed)
        self.iteration = 0
        self.device = resolve_device(cfg.get("device", "auto"))
        self.gen = make_generator(seed, self.device)
        self.out_dir = Path(cfg.get("output_dir", "runs/distill")) / "distill"
        resume_from = resume_from if resume_from is not None else self.dc.resume_from
        resume_state = self._read_state(resume_from) if resume_from is not None else None

        teacher_path = Path(cfg["teacher"])
        self.teacher_weights = teacher_path / WEIGHTS if teacher_path.is_dir() else teacher_path
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
        self.teacher_process = self._with_sampler(self.method.teacher_sampler or ("ode" if self.continuous else "ancestral"))
        for k, v in self.method.teacher_process_defaults.items():
            setattr(self.teacher_process, k, v)
        self.student_sampler = self.dc.sampler or (self.method.sampler if self.continuous else self.method.discrete_sampler)
        self.student_process = self._with_sampler(self.student_sampler)
        self.student_settings = {**self.method.student_process_defaults, **(self.dc.student_process or {})}
        for k, v in self.student_settings.items():
            if not hasattr(self.student_process, k):
                raise KeyError(f"student_process: {type(self.process).__name__} has no setting {k!r}")
            setattr(self.student_process, k, v)
        self.grid = self._student_grid()

        self.student_spec = self.method.student_spec({**self.teacher_cfg["model"], "two_time": self.method.two_time}, self)
        self.student = build_model(self.student_spec, **self.c.context).to(self.device)
        missing, unexpected = self.student.load_state_dict(self.teacher.state_dict(), strict=False)
        if unexpected or any(not k.startswith(self.method.new_param_prefixes) for k in missing):
            raise RuntimeError(f"student/teacher mismatch: missing={missing} unexpected={unexpected}")
        if self.dc.lora:
            apply_lora(self.student, **self.dc.lora)
            if self.method.two_time:
                self.student.s_embed.requires_grad_(True)
        self.student.train()
        self.roles: dict[str, Role] = {}
        self.add_role("student", self.student, ema_decay=self.dc.ema_decay)

        saved_data = resume_state.get("data") if isinstance(resume_state, dict) else None
        pool_state = saved_data if isinstance(saved_data, dict) and saved_data.get("kind") == "teacher" else None
        self.data = (TeacherPool(self, self.dc.pool_size, pool_state) if self.dc.data == "teacher"
                     else DatasetSource(self, self.dc.data))
        self.method.setup(self)
        method_spec = self.dc.method if isinstance(self.dc.method, dict) else {"type": self.dc.method}
        signature_distill = copy.deepcopy(cfg.get("distill") or {})
        # Keep `steps` in the signature: progressive/SDTT round lengths depend on it,
        # so changing steps across a resume is intentionally refused. Logging and
        # evaluation cadence do not affect the training trajectory.
        for key in ("resume_from", "save_every", "save_state_every", "log_every", "eval_every", "eval_n"):
            signature_distill.pop(key, None)
        self.trajectory_signature = _canonical_json({
            "distill": signature_distill,
            "method": method_spec,
            "student_process": {"sampler": self.student_sampler, "settings": self.student_settings},
        })
        self.trajectory_signature_sha256 = hashlib.sha256(self.trajectory_signature.encode()).hexdigest()
        if resume_from is not None:
            self.load_state(resume_from, resume_state)

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

    @staticmethod
    def _state_path(path) -> Path:
        path = Path(path)
        return path if path.suffix == ".pt" else path / "state.pt"

    @staticmethod
    def _read_state(path):
        return torch.load(DistillTrainer._state_path(path), map_location="cpu", weights_only=True)

    def _teacher_fingerprint(self) -> str:
        digest = hashlib.sha256()
        with self.teacher_weights.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def save_state(self, path) -> Path:
        """Save the complete state needed to continue the same training stream."""
        path = Path(path)
        state_path = self._state_path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        roles = {}
        for name, role in self.roles.items():
            roles[name] = {
                "model": _safe_encode(role.model.state_dict()),
                "optimizer": _safe_encode(role.optimizer.state_dict()),
                "ema": None if role.ema is None else {
                    "decay": role.ema.decay,
                    "shadow": _safe_encode(role.ema.shadow),
                },
            }
        data = None
        if isinstance(self.data, TeacherPool):
            data = {"kind": self.data.kind, "states": _safe_encode(self.data.states),
                    "x0": _safe_encode(self.data.x0), "cond": _safe_encode(self.data.cond),
                    "prompts": _safe_encode(self.data.prompts)}
        else:
            data = {"kind": self.data.kind, "state": _safe_encode(self.data.state_dict())}
        rng = {
            "python": _safe_encode(random.getstate()),
            "numpy": _safe_encode(np.random.get_state()),
            "generator": _safe_encode(self.gen.get_state()),
            "trainer": _safe_encode(self.rng.bit_generator.state),
            "torch": _safe_encode(torch.get_rng_state()),
        }
        if torch.cuda.is_available():
            rng["cuda"] = _safe_encode(torch.cuda.get_rng_state_all())
        metadata = {
            "teacher_sha256": self._teacher_fingerprint(),
            "trajectory_signature": self.trajectory_signature,
            "trajectory_signature_sha256": self.trajectory_signature_sha256,
            "data_kind": self.data.kind,
        }
        tmp_path = state_path.with_name(state_path.name + ".tmp")
        torch.save({"version": 1, "metadata": metadata, "iteration": self.iteration, "roles": roles,
                    "data": data, "method": _safe_encode(self.method.state_dict()),
                    "grid": _safe_encode(self.grid), "rng": rng}, tmp_path)
        os.replace(tmp_path, state_path)
        return path

    def load_state(self, path, state=None) -> None:
        """Load a state saved by :meth:`save_state`."""
        state = self._read_state(path) if state is None else state
        metadata = state.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint has no safe distillation metadata; cannot verify resume compatibility")
        saved_teacher = metadata.get("teacher_sha256")
        current_teacher = self._teacher_fingerprint()
        if saved_teacher != current_teacher:
            raise ValueError("teacher fingerprint mismatch: checkpoint teacher_sha256 "
                             f"{saved_teacher!r}, current teacher_sha256 {current_teacher!r}")
        saved_signature = metadata.get("trajectory_signature_sha256")
        if saved_signature != self.trajectory_signature_sha256:
            raise ValueError("trajectory config signature mismatch: checkpoint "
                             f"{saved_signature!r}, current {self.trajectory_signature_sha256!r}")
        data = state.get("data")
        if not isinstance(data, dict):
            raise ValueError("checkpoint has no distillation data state")
        saved_kind = data.get("kind", metadata.get("data_kind"))
        if saved_kind != self.data.kind:
            raise ValueError(f"data kind mismatch: checkpoint has {saved_kind!r}, trainer has {self.data.kind!r}")
        saved_roles = state["roles"]
        if set(saved_roles) != set(self.roles):
            raise ValueError(f"checkpoint roles {sorted(saved_roles)} do not match trainer roles {sorted(self.roles)}")
        for name, role in self.roles.items():
            saved = _safe_decode(saved_roles[name])
            role.model.load_state_dict(saved["model"])
            role.optimizer.load_state_dict(saved["optimizer"])
            if role.ema is None:
                if saved["ema"] is not None:
                    raise ValueError(f"checkpoint has an EMA for role {name!r}, but the trainer does not")
            else:
                if saved["ema"] is None:
                    raise ValueError(f"checkpoint has no EMA for role {name!r}")
                role.ema.decay = saved["ema"]["decay"]
                device = next(role.model.parameters()).device
                role.ema.shadow = {k: v.to(device) for k, v in saved["ema"]["shadow"].items()}
        if isinstance(self.data, TeacherPool):
            self.data._load_state(data, self.device)
        else:
            self.data.load_state_dict(_safe_decode(data.get("state", {})))
        self.method.load_state_dict(_safe_decode(state.get("method", {})), self)
        if "grid" in state:
            self.grid = _safe_decode(state["grid"]).to(self.device)
        self.iteration = int(state["iteration"])
        rng = state["rng"]
        random.setstate(_safe_decode(rng["python"]))
        np.random.set_state(_safe_decode(rng["numpy"]))
        self.rng.bit_generator.state = _safe_decode(rng["trainer"])
        torch.set_rng_state(_safe_decode(rng["torch"]))
        self.gen.set_state(_safe_decode(rng["generator"]))
        if torch.cuda.is_available() and "cuda" in rng:
            torch.cuda.set_rng_state_all(_safe_decode(rng["cuda"]))

    def train(self) -> Path:
        logger = JsonlLogger(self.out_dir / "log.jsonl")
        totals: dict[str, list[float]] = {}
        for it in range(self.iteration, self.dc.steps):
            for name in self.method.schedule(it, self):
                for k, v in self._update(name).items():
                    totals.setdefault(k, []).append(v)
            self.iteration = it + 1
            last = it + 1 == self.dc.steps
            if (it + 1) % self.dc.log_every == 0 or last:
                row = {k: float(np.mean(v)) for k, v in totals.items()}
                if self.dc.eval_every and ((it + 1) % self.dc.eval_every == 0 or last):
                    row.update(self.evaluate())
                logger.log(it + 1, row)
                totals = {}
            if self.dc.save_every and (it + 1) % self.dc.save_every == 0:
                self.export(self.out_dir / f"step_{it + 1}")
            if self.dc.save_state_every and (it + 1) % self.dc.save_state_every == 0:
                self.save_state(self.out_dir / "state")
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
        from diffusion_omni.reward.metrics import set_metrics

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
        cfg["model"] = dict(self.student_spec)
        if isinstance(cfg["process"], str):
            cfg["process"] = {"type": cfg["process"]}
        cfg["process"]["sampler"] = self.student_sampler
        cfg["process"].update(self.student_settings)
        grid = [round(float(t), 6) for t in self.grid]
        cfg["sampling"] = {"num_steps": len(grid) - 1, "guidance_scale": 1.0, "timesteps": grid}
        cfg["distilled_from"] = {"teacher": str(self.cfg["teacher"]), "method": self.dc.method,
                                 "teacher_guidance": self.dc.teacher_guidance, "steps": self.dc.steps}
        with self.eval_student() as m:
            state = merged_state_dict(m)
        return save_checkpoint(path, state, cfg)
