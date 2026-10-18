from __future__ import annotations


def _scorecard(args):
    from .scorecard import run_scorecard, save, to_markdown

    checkpoints = dict(item.split("=", 1) for item in args.ckpt or [])
    variants = {}
    for item in args.ref_variant or []:
        name, _, overrides = item.partition(":")
        variants[name] = overrides.split(",")
    rows = run_scorecard(args.ref, checkpoints, args.nfe, args.n, args.ref_steps, args.seed, args.device,
                         variants=variants)
    print(to_markdown(rows))
    if args.out:
        print(f"wrote {save(rows, args.out)}")


def register(sub) -> None:
    p = sub.add_parser("scorecard", help="compare checkpoints across step counts against a reference")
    p.add_argument("--ref", required=True, help="reference (teacher) checkpoint directory")
    p.add_argument("--ckpt", action="append", help="name=checkpoint_dir, repeatable")
    p.add_argument("--ref-variant", action="append",
                   help="tuned baseline as name:key=value[,key=value], e.g. T0.7:process.temperature=0.7")
    p.add_argument("--nfe", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("-n", type=int, default=256)
    p.add_argument("--ref-steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", help="path stem for .md and .json")
    p.set_defaults(fn=_scorecard)
