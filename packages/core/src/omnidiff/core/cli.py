"""``omnidiff <command>``. Core ships ``sample`` and ``list``; packages add commands through the
``omnidiff.commands`` entry-point group, whose targets are ``register(subparsers)`` functions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import parse_value
from .registry import COMMAND_GROUP, entry_points


def _sample(args):
    from .pipeline import Pipeline
    from .utils import to_grid, write_png

    pipe = Pipeline.from_checkpoint(args.ckpt, args.device, args.overrides)
    prompts = [parse_value(p) for p in args.prompt] if args.prompt else None
    gen = pipe.generate(prompts, n=args.n, num_steps=args.steps, guidance_scale=args.guidance, seed=args.seed)
    if isinstance(gen.samples, list):
        for p, s in zip(gen.prompts, gen.samples):
            print(f"[{p}] {s}" if p is not None else s)
        return
    out = Path(args.out or Path(args.ckpt) / "samples.png")
    write_png(out, to_grid(gen.samples), scale=args.scale)
    np.save(out.with_suffix(".npy"), gen.samples)
    print(f"prompts: {gen.prompts}\nwrote {out}")


def _list(args):
    from .registry import ALL, discover_plugins

    discover_plugins()
    for name, reg in ALL.items():
        print(f"{name:14s} {', '.join(reg)}")


def add_config_command(sub, name: str, help: str, fn):
    """A ``-c config.yaml [key=value ...]`` command, the shape every training command shares."""
    p = sub.add_parser(name, help=help)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. rl.lr=1e-4")
    p.set_defaults(fn=fn)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omnidiff")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample", help="generate from a checkpoint directory")
    p.add_argument("--ckpt", required=True)
    p.add_argument("-n", type=int, default=8)
    p.add_argument("--prompt", action="append", help="repeatable; class id or text prefix")
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device")
    p.add_argument("--out")
    p.add_argument("--scale", type=int, default=4, help="nearest-neighbour upscale for the PNG")
    p.add_argument("overrides", nargs="*")
    p.set_defaults(fn=_sample)
    sub.add_parser("list", help="show registered components").set_defaults(fn=_list)
    for ep in entry_points(COMMAND_GROUP):
        ep.load()(sub)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
