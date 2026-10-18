"""How close is a flow-map student to the teacher's true flow map, by jump size?

The MeanFlow residual is self-referential (its target uses the student's own derivative), so it can stay high
while the student still improves. This measures the student against ground truth instead: the teacher's average
velocity u*(x_t, t, s) = (x_t - X(x_t, t -> s)) / (t - s), with X integrated by a fine Euler ODE. States x_t come
from noising teacher samples. The teacher's own one-step Euler jump (u = v(x_t, t)) is the baseline, which is also
the student at initialisation.

Usage:
    uv run python scripts/flow_map_diagnose.py runs/distill/image_flow_flow_map/distill/final [more students...]
    uv run python scripts/flow_map_diagnose.py --smoothness runs/image_flow_grpo/sft/final
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from diffusion_omni.core.config import load_config
from diffusion_omni.core.pipeline import Pipeline
from diffusion_omni.core.registry import discover_plugins
from diffusion_omni.core.processes.base import bcast
from diffusion_omni.core.sampling import predict, sample

GAPS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0)]


@torch.no_grad()
def teacher_map(teacher, process, z, t, s, cond, gs, n_sub: int):
    x = z
    for k in range(n_sub):
        a = t + (s - t) * (k / n_sub)
        b = t + (s - t) * ((k + 1) / n_sub)
        x = x + bcast(b - a, x) * predict(teacher, x, a, cond, gs, process)
    return x


@torch.no_grad()
def diagnose(students: list[str], n: int = 512, n_sub: int = 64, seed: int = 0) -> list[dict]:
    discover_plugins()
    cfg = load_config(f"{students[0]}/config.yaml")
    src = cfg["distilled_from"]
    gs = src.get("teacher_guidance", 1.0)
    tpipe = Pipeline.from_checkpoint(src["teacher"], "cpu", ["process.sampler=ode"])
    teacher, process = tpipe.c.model.eval(), tpipe.c.process
    prompts = tpipe.c.prompts.sample(n, np.random.default_rng(seed))
    cond = tpipe.c.prompts.encode(prompts, "cpu")
    g = torch.Generator().manual_seed(seed + 1)
    x0 = sample(teacher, process, (n, *tpipe.c.sample_shape), cond, 32, gs, generator=g)
    t = torch.rand(n, generator=g).clamp(0.05, 1.0)
    s = t * torch.rand(n, generator=g)
    z = process.q_sample(x0, t, torch.randn(x0.shape, generator=g))
    u_star = (z - teacher_map(teacher, process, z, t, s, cond, gs, n_sub)) / bcast(t - s, z)
    models = {"teacher_euler": lambda: predict(teacher, z, t, cond, gs, process)}
    for path in students:
        m = Pipeline.from_checkpoint(path, "cpu").c.model.eval()
        models[path] = (lambda m=m: predict(m, z, t, cond, 1.0, process, s=s))
    gap = (t - s).numpy()
    norm = (u_star ** 2).flatten(1).mean(1)
    rows = []
    for name, fn in models.items():
        err = (((fn() - u_star) ** 2).flatten(1).mean(1) / norm).numpy()
        row = {"model": name, "all": float(err.mean())}
        for lo, hi in GAPS:
            sel = (gap >= lo) & (gap < hi)
            row[f"gap {lo}-{hi}"] = float(err[sel].mean()) if sel.any() else float("nan")
        rows.append(row)
    return rows


@torch.no_grad()
def smoothness(teacher_path: str, n: int = 256, hs=(0.1, 0.03, 0.01, 0.003, 0.001), seed: int = 0) -> dict:
    """Mean |dv/dt|^2 along the teacher's ODE by finite differences of step h, for t in (0.25, 1].

    For a smooth velocity field this converges as h -> 0. If it keeps growing, v jitters in t
    and exact derivatives (JVPs) see the jitter, not the flow.
    """
    discover_plugins()
    pipe = Pipeline.from_checkpoint(teacher_path, "cpu", ["process.sampler=ode"])
    model, process = pipe.c.model.eval(), pipe.c.process
    cond = pipe.c.prompts.encode(pipe.c.prompts.sample(n, np.random.default_rng(seed)), "cpu")
    g = torch.Generator().manual_seed(seed + 1)
    x0 = sample(model, process, (n, *pipe.c.sample_shape), cond, 32, generator=g)
    t = 0.25 + 0.75 * torch.rand(n, generator=g)
    z = process.q_sample(x0, t, torch.randn(x0.shape, generator=g))
    v = predict(model, z, t, cond, 1.0, process)
    out = {}
    for h in hs:
        a = (predict(model, z - h * v, t - h, cond, 1.0, process) - v) / -h
        out[h] = float((a ** 2).flatten(1).mean(1).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("students", nargs="*")
    ap.add_argument("-n", type=int, default=512)
    ap.add_argument("--n-sub", type=int, default=64)
    ap.add_argument("--smoothness", nargs="*", default=[], help="teacher checkpoints to check for jitter in t")
    args = ap.parse_args()
    for path in args.smoothness:
        print(path, " ".join(f"h={h:g}: {a:.3g}" for h, a in smoothness(path).items()))
    if not args.students:
        return
    rows = diagnose(args.students, args.n, args.n_sub)
    cols = list(rows[0])
    print("relative error ||u - u*||^2 / ||u*||^2 against the teacher's fine-ODE flow map")
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(r[c] if isinstance(r[c], str) else f"{r[c]:.4f}" for c in cols) + " |")


if __name__ == "__main__":
    main()
