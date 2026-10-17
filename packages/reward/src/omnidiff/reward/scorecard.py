"""Scorecard: checkpoints x step counts, on one fixed prompt set, in one table.

Every row is scored three ways:
  * against the reference (teacher) at full steps: teacher-relative fd / precision / recall
  * against real data, when the reference config names a dataset: data_fd / data_precision / data_recall
  * by the dataset's exact oracle and the reward, when available
The reference itself is also evaluated at each step count with its deterministic
sampler, which is the baseline a few-step student has to beat.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from omnidiff.core.pipeline import Pipeline
from omnidiff.core.registry import DATASETS, REWARDS, discover_plugins

from .metrics import RandomFeatures, set_metrics

_CONTINUOUS = ("gaussian", "flow")


def _trained_grid(pipe: Pipeline, nfe: int):
    grid = pipe.c.sampling.timesteps
    return grid if grid is not None and len(grid) - 1 == nfe else None


def run_scorecard(reference: str, checkpoints: dict[str, str] | None = None, nfes=(1, 2, 4, 8), n: int = 256,
                  ref_steps: int = 32, seed: int = 0, device: str = "cpu", reference_sampler: str | None = "auto",
                  reward: dict | str | None = "auto") -> list[dict]:
    discover_plugins()
    ref_pipe = Pipeline.from_checkpoint(reference, device)
    cfg = ref_pipe.c.cfg
    if reference_sampler == "auto":
        reference_sampler = "ode" if cfg["process"].get("type") in _CONTINUOUS else None
    if reference_sampler:
        ref_pipe = Pipeline.from_checkpoint(reference, device, [f"process.sampler={reference_sampler}"])
    ctx = ref_pipe.c.context

    prompts = ref_pipe.c.prompts.sample(n, np.random.default_rng(seed))
    ref_samples = ref_pipe.generate(prompts, num_steps=ref_steps, seed=seed).samples
    dataset = None
    if isinstance(cfg.get("sft"), dict) and cfg["sft"].get("dataset"):
        dataset = DATASETS.build(cfg["sft"]["dataset"], **ctx)
    real = None
    if dataset is not None:
        real = ref_pipe.c.codec.decode(dataset.sample(n, np.random.default_rng(seed + 1))[0])
    if reward == "auto":
        reward = (cfg.get("rl") or {}).get("reward")
    reward_fn = REWARDS.build(reward, **ctx) if reward else None
    features = None if isinstance(ref_samples, list) else RandomFeatures(int(np.prod(np.asarray(ref_samples).shape[1:])))

    def score(name, nfe, pipe, samples, seconds):
        row = {"model": name, "nfe": nfe}
        row.update(set_metrics(samples, prompts, ref_samples, features))
        if real is not None and not isinstance(samples, list):
            row.update({k: v for k, v in set_metrics(samples, prompts, real, features, "data_").items() if k != "diversity"})
        if dataset is not None and hasattr(dataset, "oracle"):
            row.update(dataset.oracle(samples, prompts))
        if reward_fn is not None:
            row["reward"] = float(np.mean(reward_fn(samples, prompts)))
        row["ms_per_sample"] = 1000 * seconds / len(prompts)
        return row

    rows = [score(f"reference@{ref_steps}", ref_steps, ref_pipe, ref_samples, float("nan"))]
    entries = [("reference", ref_pipe)] + [(k, Pipeline.from_checkpoint(v, device)) for k, v in (checkpoints or {}).items()]
    for name, pipe in entries:
        for nfe in nfes:
            grid = _trained_grid(pipe, nfe)
            start = time.perf_counter()
            gen = pipe.generate(prompts, num_steps=None if grid else nfe, timesteps=grid, seed=seed + 2)
            rows.append(score(name, nfe, pipe, gen.samples, time.perf_counter() - start))
    return rows


def to_markdown(rows: list[dict]) -> str:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    fmt = lambda v: "" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:.4g}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def save(rows: list[dict], out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    out.with_suffix(".md").write_text(to_markdown(rows) + "\n")
    return out.with_suffix(".md")
