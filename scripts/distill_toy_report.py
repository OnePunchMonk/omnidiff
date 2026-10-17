"""Score every toy distillation run against its teacher and write docs/results/distillation_toy.md.

Prerequisites (CPU, minutes each):
    uv run omnidiff sft -c configs/<teacher>.yaml           for the four teachers
    uv run omnidiff distill -c configs/distill/<name>.yaml  for each student below
Usage:
    uv run python scripts/distill_toy_report.py [-n 512]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omnidiff.reward.scorecard import run_scorecard, save, to_markdown

GROUPS = {
    "image_flow_grpo": ["image_flow_kd", "image_flow_consistency", "image_flow_flow_map", "image_flow_dmd2",
                        "image_flow_dmd2_gan", "image_flow_rcm"],
    "image_ddpm_ddpo": ["image_ddpm_dmd2"],
    "video_flow_grpo": ["video_flow_dmd2", "video_flow_flow_map"],
    "text_masked_grpo": ["text_masked_kd", "text_masked_sdtt"],
}
# Tuned baselines: the teacher with a sharper or different sampler, at the same step counts.
VARIANTS = {
    "image_flow_grpo": {"renoise": ["process.sampler=renoise"], "shift3": ["process.shift=3.0"]},
    "image_ddpm_ddpo": {"renoise": ["process.sampler=renoise"]},
    "video_flow_grpo": {"renoise": ["process.sampler=renoise"], "shift3": ["process.shift=3.0"]},
    "text_masked_grpo": {"T0.8": ["process.temperature=0.8"], "T0.6": ["process.temperature=0.6"]},
}
DROP = {"colour_err"}  # kept in the JSON, left out of the doc tables for width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=512)
    ap.add_argument("--nfe", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", default="docs/results/distillation_toy.md")
    args = ap.parse_args()
    sections = []
    for teacher, students in GROUPS.items():
        ckpts = {s: f"runs/distill/{s}/distill/final" for s in students if Path(f"runs/distill/{s}/distill/final").exists()}
        rows = run_scorecard(f"runs/{teacher}/sft/final", ckpts, args.nfe, args.n, variants=VARIANTS.get(teacher))
        save(rows, f"runs/scorecards/{teacher}")
        rows = [{k: v for k, v in r.items() if k not in DROP} for r in rows]
        sections.append(f"### Teacher `{teacher}`\n\n{to_markdown(rows)}\n")
        print(sections[-1], flush=True)
    Path(args.out).write_text("\n".join(sections))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
