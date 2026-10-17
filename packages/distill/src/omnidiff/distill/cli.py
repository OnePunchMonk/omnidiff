from __future__ import annotations

from omnidiff.core.cli import add_config_command
from omnidiff.core.config import load_config


def _distill(args):
    from .trainer import DistillTrainer

    path = DistillTrainer(load_config(args.config, args.overrides)).train()
    print(f"saved {path}")


def _export(args):
    from .export import export_bundle

    print(f"wrote {export_bundle(args.ckpt, args.out)}")


def register(sub) -> None:
    add_config_command(sub, "distill", "few-step distillation of a teacher checkpoint", _distill)
    p = sub.add_parser("export", help="write a framework-neutral student bundle from a checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_export)
