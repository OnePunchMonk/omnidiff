# Reference: diffusion-omni-distill

Package: `packages/distill`, import name `diffusion_omni.distill`. Depends on diffusion-omni-core, diffusion-omni-data, diffusion-omni-reward. Optional extra `export` adds `safetensors`.

Scope and method landscape: [docs/scope/distillation.md](../scope/distillation.md). Toy results: [docs/results/distillation_toy.md](../results/distillation_toy.md).

## Purpose

Few-step students from a trained teacher checkpoint, for text, image and video. A student is an ordinary diffusion-omni checkpoint whose config carries its sampler and timestep grid, so `Pipeline`, `sample`, `rl` and `scorecard` use it unchanged.

## Public API

| module | names |
|---|---|
| `trainer.py` | `DistillConfig`, `DistillTrainer(cfg, resume_from=None).train() -> Path`, `EMA`, `save_state`, `load_state`, `add_role(name, module, lr, betas, ema_decay)`, `teacher_pred`, `student_pred`, `prompt_cond`, `student_rollout`, `export(path)` |
| `methods/base.py` | `Distiller` (base class), `per_sample_mean` |
| `methods/*.py` | one class per method (see table below) |
| `export.py` | `export_bundle(ckpt, out) -> Path`, `load_bundle(path, device=None) -> Pipeline`, `validate_sampler(meta)` |

## CLI

### `diffusion-omni distill`

```
usage: diffusion-omni distill [-h] -c CONFIG [overrides ...]
```

Prints `saved <path>`. Output: `<output_dir>/distill/final` (the EMA student, LoRA merged, with `config.yaml` setting `process.sampler`, `sampling.timesteps` and `distilled_from`).

### `diffusion-omni export`

```
usage: diffusion-omni export [-h] --ckpt CKPT --out OUT
```

Writes a framework-neutral bundle to `--out`: `model.safetensors` (or `model.pt` without safetensors), `model.json`, `sampler.json`, `README.md`, and `resources/` if needed. The output directory must not be the checkpoint itself or an existing diffusion-omni checkpoint. Guide: [export](../guides/export.md).

## Config keys

Top level: `teacher` (path to the teacher checkpoint directory), plus the core keys. The `distill:` section is `DistillConfig`:

| key | default |
|---|---|
| `method` | required: `{type: <DISTILLERS name>, ...method kwargs}` |
| `student_steps` | none (process default spacing) |
| `student_timesteps` | none (explicit grid; overrides `student_steps`) |
| `sampler` | from the method |
| `student_process` | none (process settings for the student only, exported with it) |
| `teacher_guidance` | 1.0 (CFG applied to every teacher prediction; the student is exported with guidance 1) |
| `data` | `teacher` (a pool of teacher samples), or any `DATASETS` spec |
| `pool_size` | 2048 |
| `teacher_steps` | 32 |
| `steps` | 2000 (student updates) |
| `batch_size` | 32 |
| `lr` | 1e-4 |
| `betas` | `[0.9, 0.999]` |
| `weight_decay`, `max_grad_norm` | 0.0, 1.0 |
| `ema_decay` | 0.99 (null disables EMA) |
| `lora` | null |
| `log_every`, `eval_every`, `eval_n`, `save_every` | 50, 0, 128, 0 |
| `save_state_every` | 0 (save resumable state) |
| `resume_from` | none |

Method kwargs (from `method.type` constructors):

| `method.type` | sampler (default) | kwargs (defaults) |
|---|---|---|
| `kd` | `ode` / ancestral | none |
| `dmd2` | `renoise` | `fake_updates` (5), `tau_min`, `tau_max`, `gan_weight` (0.0), `backward_sim` (true), `fake_lr`, `last_stage_prob` |
| `dmd2_reward` | `renoise` | `reward` (required), `reward_weight` (1.0), `mode` (`gradient` or `advantage`), `group_size` (4), plus `dmd2` kwargs |
| `consistency` | `renoise` | `num_intervals` (32), `target` (`stopgrad` or `ema`), `target_decay`, `distance` (`pseudo_huber` or `l2`), `weighting` (`inverse_dt`) |
| `flow_map` | `ode` (two-time) | `equal_ratio` (0.25), `s_mode` (`uniform` or `zero`), `derivative` (`jvp` or `fd`), `fd_eps`, `time_sampling`, `adaptive_p`, `adaptive_c` |
| `rcm` | `renoise` | `dmd_weight` (0.1), `fake_updates` (5), `s_mode` (`zero`), `equal_ratio`, `derivative`, `tau_min`, `tau_max`, `fake_lr` |
| `sdtt` | ancestral (masked) | `start_steps` (32), `rounds` (3), `sub_steps` (2) |
| `progressive` | per class | `start_steps` (16), `rounds` (3), `weighting` (`truncated_snr`) |
| `dcd` | per class | `sub_steps` (2), `boundary_prob` (0.25), `target_decay` (0.95), `t_min` (0.05) |
| `coupled_kd` | `coupled` (masked) | `noise` (`gumbel` or `uniform`), `noise_features` (64) |

Constraints: `flow_map` warns when the teacher's `model.time_scale` is above 100 (the default is 1000). Train the teacher with `model.time_scale: 10`. A resume is refused when the training signature changes (method, `distill` settings including `steps`, and the student process). Logging and evaluation cadence (`resume_from`, `save_every`, `save_state_every`, `log_every`, `eval_every`, `eval_n`) can change.

## Shipped recipes

`configs/distill/` holds one file per method and modality: `image_ddpm_*`, `image_flow_*`, `video_flow_*`, `text_masked_*`. Examples: `image_flow_dmd2.yaml`, `image_flow_dmd2_reward.yaml`, `image_flow_flow_map.yaml`, `text_masked_coupled_kd.yaml`, `text_masked_sdtt.yaml`, `video_flow_rcm.yaml`. Check the file names with `ls configs/distill`.

## Registries

`DISTILLERS`: `kd`, `dmd2`, `dmd2_reward`, `consistency`, `flow_map`, `rcm`, `progressive`, `sdtt`, `coupled_kd`, `dcd`.

## Extension points

Subclass `diffusion_omni.distill.methods.base.Distiller`, register it in `DISTILLERS`, and declare:

- `processes` (which process types it supports), `sampler`, `discrete_sampler`, `two_time`, and `teacher_process_defaults` / `student_process_defaults`.
- `loss(role, tr) -> (loss, metrics)`. Use the trainer helpers `teacher_pred`, `student_pred`, `student_rollout`, `prompt_cond`, and `tr.data.batch`.
- Optionally `setup`, `schedule(it, tr)` (the roles to update, in order), `after_update`, `state_dict` / `load_state_dict`, and `student_spec`.

Roles: the trainer holds a frozen teacher and a student, plus roles added with `tr.add_role(name, module, lr=...)`. Each role has its own optimiser, and only the role being updated is trainable.

Tests: `packages/distill/tests/test_distill.py` (method properties, `test_distill_config_end_to_end`, `test_renoise_student_is_rl_tunable`), `test_dmd2_reward.py`, `test_dcd.py`, `test_export.py`.
