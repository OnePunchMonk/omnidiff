# Reference: diffusion-omni-reward

Package: `packages/reward`, import name `diffusion_omni.reward`. Depends on diffusion-omni-core and diffusion-omni-data.

## Purpose

Reward functions for RL and distillation, set-level metrics, aggregation over seeds, and the distillation scorecard.

## Public API

| module | names |
|---|---|
| `rewards.py` | `ChannelDominance(channel=0)`, `Brightness(target=None)`, `TemporalSmoothness()`, `CharFraction(chars="aeiou", completion_only=True)`, `Contains(substring)`, `Composite(terms, **context)` |
| `metrics.py` | `RandomFeatures(in_dim, out_dim=64, seed=0)`, `frechet_distance(f1, f2)`, `precision_recall(real, fake, k=3)`, `within_prompt_diversity(samples, prompts)`, `set_metrics(samples, prompts, reference=None, features=None, prefix="")` |
| `aggregate.py` | `aggregate(rows, ddof=1)`, `to_markdown(agg, expected=None)` |
| `scorecard.py` | `run_scorecard(reference, checkpoints, nfes, n, ref_steps, seed, device, reference_sampler, reward, variants)`, `to_markdown(rows)`, `save(rows, out)` |

Reward call semantics: `reward(samples, prompts) -> np.ndarray` of shape `(B,)`. Samples are decoded (numpy in [0, 1] for pixels, `list[str]` for text). A reward that is differentiable also defines `torch_call(x, prompts=None)` and sets `differentiable = True`. `dmd2_reward` uses it.

## CLI

### `diffusion-omni scorecard`

```
usage: diffusion-omni scorecard [-h] --ref REF [--ckpt CKPT]
                          [--ref-variant REF_VARIANT] [--nfe NFE [NFE ...]]
                          [-n N] [--ref-steps REF_STEPS] [--seed SEED]
                          [--device DEVICE] [--out OUT]
```

- `--ref`: teacher checkpoint directory (required).
- `--ckpt name=dir`: repeatable. Students to score.
- `--ref-variant name:key=value[,key=value]`: a tuned teacher baseline, e.g. `T0.7:process.temperature=0.7`.
- `--nfe`: step counts to evaluate (default `1 2 4 8`). The trained grid is used when the count matches it.
- `-n`: samples per row (default 256). `--ref-steps`: teacher steps for the reference row (default 32).
- `--out`: path stem; writes `<stem>.md` and `<stem>.json`.

Columns: diversity, teacher-relative `fd`, precision, recall, data-relative `data_fd`, `data_precision`, `data_recall` (when the reference config names a dataset), dataset oracle metrics, reward, and `ms_per_sample`.

Guide: [eval-scorecard](../guides/eval-scorecard.md).

## Config keys

Rewards are configured under `rl.reward` and, for distillation, under `distill.method.reward` (`dmd2_reward`):

```yaml
rl:
  reward: {type: channel_dominance, channel: 0}

rl:
  reward:
    type: composite
    terms:
      - {weight: 1.0, reward: {type: brightness}}
      - {weight: 1.0, reward: {type: temporal_smoothness}}
```

`dmd2_reward` takes its reward under `distill.method.reward`. `diffusion-omni scorecard` uses the reference config's `rl.reward`. `run_scorecard(..., reward=...)` accepts an explicit spec instead.

## Registries

`REWARDS`: `channel_dominance`, `brightness`, `temporal_smoothness`, `char_fraction`, `contains`, `composite`.

## Extension points

Implement `__call__(samples, prompts)` and register it in `REWARDS`, or use `type: my_pkg.rewards:MyReward`. Add `torch_call` if the reward should carry gradients into distillation.

Tests: `packages/reward/tests/test_metrics.py`, `packages/reward/tests/test_aggregate.py`, `packages/reward/tests/test_distill_report.py`.
