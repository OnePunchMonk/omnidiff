# Guide: distill

Few-step distillation of a trained teacher. The student keeps the teacher's architecture and is exported as a normal diffusion-omni checkpoint, so the same `sample`, `rl` and `scorecard` commands work on it.

Needs a teacher checkpoint (see [sft](sft.md)).

## Tiny run (CPU)

```bash
uv run diffusion-omni distill -c configs/distill/image_flow_dmd2.yaml device=cpu teacher=runs/guide/img/sft/final distill.steps=20 distill.pool_size=32 distill.teacher_steps=4 distill.batch_size=4 distill.eval_every=0 output_dir=runs/guide/dmd2
uv run diffusion-omni sample --ckpt runs/guide/dmd2/distill/final --prompt 0 --prompt 1 --device cpu --out runs/guide/dmd2/samples.png
```

Output: `saved runs/guide/dmd2/distill/final`. The student's `config.yaml` holds its sampler, its timestep grid, and `distilled_from`.

## Full run

```bash
uv run diffusion-omni distill -c configs/distill/image_flow_dmd2.yaml
```

This uses `teacher: runs/image_flow_grpo/sft/final`, 4 student steps and 600 updates.

## Choosing a method

Pick a file in `configs/distill/` or set `distill.method`:

| `method.type` | use it for |
|---|---|
| `kd` | the baseline: regress teacher-ODE states onto their endpoints |
| `dmd2` | few-step continuous students (image, video); data-free; two time-scale updates |
| `dmd2_reward` | DMD2 with a reward term (`reward`, `reward_weight`, `mode`) |
| `consistency` | consistency distillation (flow or Gaussian) |
| `flow_map` | average-velocity student; needs a teacher trained with `model.time_scale: 10` |
| `rcm` | flow map with a score regulariser |
| `progressive`, `dcd` | progressive halving and discrete consistency |
| `sdtt`, `coupled_kd` | masked text diffusion (`coupled_kd` samples a student that reproduces the teacher's noise-coupled samples) |

Method kwargs and defaults are in the [distill reference](../reference/distill.md). Toy results comparing methods are in [docs/results/distillation_toy.md](../results/distillation_toy.md).

## Keys you will change

| key | what it does |
|---|---|
| `teacher` | teacher checkpoint directory (top level) |
| `distill.student_steps` or `distill.student_timesteps` | student grid |
| `distill.steps`, `distill.batch_size`, `distill.lr` | training length and size |
| `distill.data` | `teacher` (sample a pool from the teacher) or a dataset spec |
| `distill.teacher_guidance` | CFG applied to every teacher prediction; the student is exported with guidance 1 |
| `distill.save_state_every`, `distill.resume_from` | resumable state |

Reference: [distill](../reference/distill.md).
