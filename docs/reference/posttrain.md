# Reference: diffusion-omni-posttrain

Package: `packages/posttrain`, import name `diffusion_omni.posttrain`. Depends on diffusion-omni-core, diffusion-omni-data, diffusion-omni-reward.

## Purpose

Supervised denoising training (`sft`) and online RL (`rl`) on the denoising MDP. The RL objective is a PPO-clipped surrogate per denoising step, with an optional exact per-step KL to a reference policy.

## Public API

| module | names |
|---|---|
| `sft.py` | `SFTConfig`, `SFTTrainer(cfg, components=None).train() -> Path` |
| `rl/trainer.py` | `RLConfig`, `RLTrainer(cfg, components=None).train() -> Path`, `Rollout` |
| `rl/algorithms.py` | `compute_advantages(rewards, groups, mode, eps, clip)`, `LossOutput`, `PolicyGradient`, `GRPO`, `DDPO` |

`LossOutput` holds `loss` and a `metrics` dict (`pg_loss`, `ratio`, `clip_frac`, and `kl` when used).

## CLI

### `diffusion-omni sft`

```
usage: diffusion-omni sft [-h] -c CONFIG [overrides ...]
```

### `diffusion-omni rl`

```
usage: diffusion-omni rl [-h] -c CONFIG [overrides ...]
```

Both read the whole config, so `sft.*` and `rl.*` overrides work from either command. Each prints `saved <path>`. Output: `<output_dir>/sft/final` or `<output_dir>/rl/final` (each with `config.yaml` and `model.pt`), and `<output_dir>/sft/log.jsonl` or `<output_dir>/rl/log.jsonl`.

## Config keys

`sft`:

| key | default |
|---|---|
| `dataset` | required |
| `init_from` | none |
| `steps` | 1000 |
| `batch_size` | 32 |
| `lr` | 3e-4 |
| `weight_decay` | 0.0 |
| `warmup_steps` | 100 |
| `cond_dropout` | 0.1 (label dropout, so the unconditional branch trains for CFG) |
| `max_grad_norm` | 1.0 |
| `log_every` | 50 |
| `save_every` | 0 |

`rl`:

| key | default |
|---|---|
| `reward` | required |
| `algorithm` | `{type: grpo}` |
| `init_from` | none (set it to the SFT checkpoint) |
| `lora` | null (full fine-tune); `{rank, alpha, dropout, targets}` for LoRA |
| `epochs` | 100 |
| `prompts_per_epoch` | 8 |
| `group_size` | 4 (samples per prompt) |
| `rollout_batch_size` | 32 |
| `train_batch_size` | 16 |
| `inner_epochs` | 1 |
| `timestep_fraction` | 1.0 (fraction of stochastic steps trained per minibatch) |
| `lr`, `weight_decay`, `max_grad_norm`, `save_every` | 1e-4, 0.0, 1.0, 0 |

`algorithm` keys: `clip_range` (1e-4 for `grpo` and `ddpo` by default), `kl_coef` (0.0), `adv_mode` (`group` for GRPO, `batch` for DDPO), `adv_clip` (5.0).

Constraints: the sampler must be stochastic. An ODE configuration (`eta: 0`, `noise_level: 0`) is rejected with an error. The last deterministic step to t = 0 has no log-probability and is not trained.

## Registries

- `ALGORITHMS`: `grpo` (group-relative advantages, for images, video and text), `ddpo` (batch-normalised advantages, no KL by default).

Rewards come from `REWARDS` (reward package). Processes and models come from core.

## Extension points

- **RL objective**: subclass `PolicyGradient` or implement `advantages(rewards, groups)`, `loss(log_prob, old_log_prob, advantages, kl)` returning `LossOutput`, and a `kl_coef` attribute. Register in `ALGORITHMS`.
- **Reward**: see the reward reference.

Tests: `packages/posttrain/tests/test_posttrain_e2e.py` (SFT loss is finite, rollout log-probs are on-policy, SFT then RL then sample, group advantages, plugin components from config).
