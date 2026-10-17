"""omnidiff sft|rl|sample|list"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import load_config, parse_value


def _train(args, trainer_cls):
    cfg = load_config(args.config, args.overrides)
    path = trainer_cls(cfg).train()
    print(f"saved {path}")


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
    from .registry import ALL

    for name, reg in ALL.items():
        print(f"{name:14s} {', '.join(reg)}")


def main(argv=None):
    from .rl.trainer import RLTrainer
    from .sft import SFTTrainer

    ap = argparse.ArgumentParser(prog="omnidiff")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, cls in (("sft", SFTTrainer), ("rl", RLTrainer)):
        p = sub.add_parser(name, help=f"{name} training from a config")
        p.add_argument("-c", "--config", required=True)
        p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. rl.lr=1e-4")
        p.set_defaults(fn=lambda a, cls=cls: _train(a, cls))
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
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
