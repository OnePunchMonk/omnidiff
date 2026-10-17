# omnidiff-distill

Few-step distillation for text, image and video diffusion, built on `omnidiff-core`. The scope and the method landscape are in [docs/scope/distillation.md](../../docs/scope/distillation.md).

```bash
uv run omnidiff sft     -c configs/image_flow_grpo.yaml                 # teacher
uv run omnidiff distill -c configs/distill/image_flow_dmd2.yaml         # 4-step student
uv run omnidiff scorecard --ref runs/image_flow_grpo/sft/final \
    --ckpt dmd2=runs/distill/image_flow_dmd2/distill/final --nfe 1 2 4 8 --out runs/scorecard
uv run omnidiff sample --ckpt runs/distill/image_flow_dmd2/distill/final --prompt 0 --prompt 1
```

The student checkpoint is a normal omnidiff checkpoint. Its config carries the sampler and timestep grid it was trained for, so `Pipeline`, `sample`, `rl` and `scorecard` all use it unchanged.

## Methods

| `method.type` | processes | student sampler | idea | needs |
|---|---|---|---|---|
| `kd` | flow, gaussian, masked | `ode` / ancestral | regress teacher-ODE states on the student grid onto the trajectory endpoint (cross-entropy on final tokens for text) | teacher pool |
| `dmd2` | flow, gaussian | `renoise` | x0 pushed along (real − fake score); fake score trained online on student samples; two time-scale updates, backward simulation, optional GAN head | prompts; real data only for the GAN |
| `consistency` | flow, gaussian | `renoise` | f(x_hi, t_hi) = sg f(teacher ODE step to t_lo); iCT stop-gradient target, pseudo-Huber, 1/Δt weighting | x0 (pool or dataset) |
| `flow_map` | flow | `ode` (two-time) | average velocity u(x, t, s) trained with the MeanFlow identity using the teacher's velocity; d/dt by forward-mode JVP or finite difference; `s_mode: zero` gives a continuous-time consistency model | x0 (pool or dataset) |
| `sdtt` | masked | ancestral | rounds of "student step = 2 teacher steps"; targets are the teacher's distribution at the sub-step where each token was decoded | x0 (dataset or pool) |

## The trainer

- **Roles.** The trainer holds a frozen teacher and a `student` initialised from it, plus any roles a method adds with `tr.add_role(name, module, lr=...)`. Each role has its own optimiser. While one role updates, every other role is frozen. Gradients still flow *through* frozen networks to their inputs, which is how DMD2's generator loss reaches the student.
- **Schedule.** `method.schedule(it, tr)` returns the role updates for one iteration. For DMD2 that is `["fake"] * fake_updates + ["student"]`.
- **Data.** `distill.data: teacher` builds a pool of teacher samples with the teacher's deterministic sampler. It also records teacher states on the student's grid. Alternatively, any `DATASETS` spec can be the data.
- **Teacher guidance.** Every teacher prediction uses `teacher_guidance`. The student is exported with guidance 1, so CFG is distilled in.
- **Export.** The EMA student, with LoRA merged, is saved together with a config that sets `process.sampler`, `sampling.timesteps` and provenance (`distilled_from`).

## Few-step students are RL-able

The `renoise` sampler (predict x0, then re-noise to the next time) is a Gaussian policy with exact log-probs. DMD2 and consistency students can therefore be RL-tuned with `omnidiff rl` straight from their checkpoint:

```bash
uv run omnidiff rl -c runs/distill/image_flow_dmd2/distill/final/config.yaml \
    rl.init_from=runs/distill/image_flow_dmd2/distill/final output_dir=runs/rl_on_student
```

`tests/test_distill.py::test_renoise_student_is_rl_tunable` checks this.

## Correctness tests

| property | test |
|---|---|
| DMD2's generator gradient equals (x0_fake − x0_real)·w / numel | `test_dmd2_generator_gradient_is_the_score_difference` |
| a flow-map student starts as the teacher's Euler sampler | `test_flow_map_student_starts_as_teacher_euler` |
| the forward-mode JVP of the average velocity matches a finite difference | `test_flow_map_jvp_matches_finite_difference` |
| SDTT targets are distributions over clean tokens, and rounds snapshot the EMA student | `test_sdtt_targets_are_distributions_and_rounds_advance` |
| the consistency boundary f(x, 0) = x holds for the student | `test_consistency_boundary_target_is_the_teacher_step` |
| every shipped config trains, exports and samples | `test_distill_config_end_to_end` |

## Adding a method

Subclass `omnidiff.distill.methods.base.Distiller` and register it under `DISTILLERS`. Then declare `processes`, the default `sampler`, `two_time`, and implement `loss(role, tr)`, plus `setup` / `schedule` / `after_update` if needed. Use the trainer helpers `teacher_pred`, `student_pred`, `student_rollout`, `prompt_cond` and `data.batch`.
