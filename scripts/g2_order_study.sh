#!/usr/bin/env bash
# Does the order of RL and distillation matter? Toy study on the image flow teacher, CPU only.
#   A: SFT teacher -> Flow-GRPO (10-step SDE)                     runs/image_flow_grpo/rl/final
#   B: SFT teacher -> DMD2 4-step student -> GRPO on the student  runs/g2/distill_then_rl/rl/final
#   C: SFT teacher -> Flow-GRPO -> DMD2 4-step student            runs/distill/image_flow_dmd2_from_rl/distill/final
# All are scored at 4 NFE against the SFT teacher: reward, precision/recall, diversity, oracle.
set -euo pipefail
STUDENT=runs/distill/image_flow_dmd2/distill/final
uv run diffusion-omni rl -c "$STUDENT/config.yaml" device=cpu "rl.init_from=$STUDENT" output_dir=runs/g2/distill_then_rl \
  rl.epochs=40 rl.lora='{rank: 8, alpha: 16}'
uv run diffusion-omni distill -c configs/distill/image_flow_dmd2_from_rl.yaml device=cpu
uv run diffusion-omni scorecard --ref runs/image_flow_grpo/sft/final -n 512 --nfe 4 10 --out runs/scorecards/g2_order \
  --ckpt A_rl_teacher=runs/image_flow_grpo/rl/final \
  --ckpt B_distill_then_rl=runs/g2/distill_then_rl/rl/final \
  --ckpt C_rl_then_distill=runs/distill/image_flow_dmd2_from_rl/distill/final \
  --ckpt dmd2_student=$STUDENT
