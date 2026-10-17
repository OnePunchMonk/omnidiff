# omnidiff

A diffusion monorepo for **language, image and video**, built in plain PyTorch and NumPy. It covers the layers that SGLang Diffusion (serving) and miles-diffusion (large-scale RL) leave open, on one shared core.

Today it has the core, RL/SFT post-training, and few-step distillation (KD, DMD2, consistency, flow maps, rCM, SDTT and noise-conditioned text distillation), plus a scorecard that compares students and teachers across step counts. Every component plugs in through a registry.

`omnidiff-core` depends only on `torch`, `numpy` and `pyyaml`. It has no `diffusers`, `transformers`, `accelerate` or `trl`, so everything that runs is in this repo.

## The one idea

Every diffusion sampler is an MDP. The state is `(x_t, t, cond)`, the action is `x_next`, and the reward comes at the end. If each reverse step has an exact log-probability, the same PPO/GRPO code trains all three modalities:

| process (`process.type`) | modality | reverse step = policy | log π(x_next \| x_t) |
|---|---|---|---|
| `masked`: absorbing discrete diffusion (MDLM / LLaDA) | text | unmask each masked token w.p. (t−s)/t, draw it from the predicted categorical | exact: categorical + Bernoulli terms |
| `gaussian`: VP diffusion, ε-prediction, DDIM(η) | image, video | Gaussian, η > 0 (η = 1 ≈ DDPM, the DDPO policy) | exact Gaussian |
| `flow`: rectified flow, v-prediction | image, video | marginal-preserving reverse SDE from Flow-GRPO | exact Gaussian |

`DiffusionProcess.transition(pred, x_t, t, t_next, x_next=None)` samples a step. Pass `x_next` and it scores a stored action instead. Rollouts and RL updates go through that same call, so the ratio for a fresh rollout is exactly 1 (`tests/test_e2e.py::test_rollout_log_probs_are_on_policy`).

## Monorepo layout

Goal, layer map and roadmap: **[docs/GOAL_AND_PLAN.md](docs/GOAL_AND_PLAN.md)**. Distillation scope and status: **[docs/scope/distillation.md](docs/scope/distillation.md)**.

```
packages/
  core/       omnidiff-core       omnidiff.core       processes, models, sampling, inference, CLI   (torch + numpy only)
  posttrain/  omnidiff-posttrain  omnidiff.posttrain  SFT, GRPO, DDPO                               done
  distill/    omnidiff-distill    omnidiff.distill    KD, DMD2, consistency, flow map, rCM, SDTT, coupled KD  v0
  reward/     omnidiff-reward     omnidiff.reward     rewards, set metrics, scorecard               seed+
  data/       omnidiff-data       omnidiff.data       datasets -> data engine                       seed
configs/      one YAML per recipe (sft + rl + sampling sections)
docs/         goal, plan, per-layer scope docs
examples/     custom_plugin.py
```

Every package is a separate distribution under the shared `omnidiff.*` namespace. A package plugs in by declaring entry points: `omnidiff.plugins` registers its components, and `omnidiff.commands` adds CLI subcommands. Installing a package is enough to make its components buildable from config.

Inside `omnidiff.core`:

```
registry.py          string -> constructor registries, context injection, `pkg.mod:Class` specs, entry-point discovery
config.py            YAML + dotted CLI overrides
build.py             config -> Components (codec, process, model, prompts)
processes/           gaussian.py  flow.py  masked.py   (forward noising, training loss, policy step, step KL)
nn/                  adapters (tokens | image_patch | video_patch), DiT backbone, conditioners, denoiser, LoRA
sampling.py          predict (with CFG), step, sample -> Trajectory
codecs.py            pixel | text (byte tokenizer)     data <-> process tensors (a VAE goes here)
prompts.py           none | class | text               prompt sampling + encoding to cond
pipeline.py          inference API
testing.py           tiny_config(): any config shrunk to seconds on CPU
```

## Quickstart

```bash
uv sync                                  # installs every package editable into .venv
uv run pytest                            # all packages, a few seconds on CPU

# each config runs the whole loop on a laptop CPU in minutes
uv run omnidiff sft -c configs/image_flow_grpo.yaml
uv run omnidiff rl  -c configs/image_flow_grpo.yaml
uv run omnidiff sample --ckpt runs/image_flow_grpo/rl/final --prompt 0 --prompt 1 --prompt 2 --prompt 3

uv run omnidiff sft -c configs/text_masked_grpo.yaml
uv run omnidiff rl  -c configs/text_masked_grpo.yaml
uv run omnidiff sample --ckpt runs/text_masked_grpo/rl/final --prompt "the cat" --prompt "a dog"

# few-step distillation of a trained teacher, then compare across step counts
uv run omnidiff distill -c configs/distill/image_flow_dmd2.yaml
uv run omnidiff scorecard --ref runs/image_flow_grpo/sft/final --ckpt dmd2=runs/distill/image_flow_dmd2/distill/final

uv run omnidiff list                     # everything registered by installed packages
```

Override any key from the command line: `uv run omnidiff rl -c configs/video_flow_grpo.yaml rl.lr=1e-4 rl.lora=null device=cuda`.

From Python:

```python
from omnidiff.core import Pipeline
pipe = Pipeline.from_checkpoint("runs/text_masked_grpo/rl/final")
gen = pipe.generate(prompts=["the sun"], num_steps=32, seed=0)
print(gen.samples)          # decoded: list[str] for text, float numpy in [0, 1] for pixels
```

## Shipped configs

| config | modality | process | algorithm | reward |
|---|---|---|---|---|
| `text_masked_grpo` | text, 32 bytes | masked | GRPO, full FT | vowel fraction of the completion |
| `image_flow_grpo` | image 3×16×16, class-cond | flow SDE | Flow-GRPO, LoRA | red-channel dominance |
| `image_ddpm_ddpo` | image 3×16×16, class-cond | gaussian η=1 | DDPO, full FT | brightness |
| `video_flow_grpo` | video 3×4×16×16, direction-cond | flow SDE | Flow-GRPO, LoRA, KL to base | brightness + temporal smoothness |

The toy data and rewards exist so the full loop can be checked in minutes. They are not benchmarks.

## Plugging things in

Every config entry is `{type: name, ...kwargs}`. A `type` can be a registered name or any import path (`my_pkg.rewards:Aesthetic`). Modules listed under `plugins:` are imported first, so their `@REGISTRY.register(...)` decorators run. Installed packages with an `omnidiff.plugins` entry point load automatically.

A **new layer** is a new `packages/<name>/` with its own `pyproject.toml`. It depends on `omnidiff-core` and declares entry points: `omnidiff.plugins` for components and `omnidiff.commands` for CLI subcommands. `packages/posttrain/pyproject.toml` is the template.

Constructors receive shared **context** by naming it in their signature: `dim`, `sample_shape`, `codec`, `mask_id`, `vocab_size`. That's how `TokenAdapter` gets the vocabulary from the tokenizer without the config repeating it.

| to add | implement | register in |
|---|---|---|
| reward model | `__call__(samples, prompts) -> np.ndarray (B,)` | `REWARDS` |
| RL objective | `advantages(rewards, groups)`, `loss(logp, old_logp, adv, kl) -> LossOutput`, `kl_coef` | `ALGORITHMS` (posttrain) |
| diffusion process | `DiffusionProcess` (timesteps, prior, training_loss, transition, kl) | `PROCESSES` |
| modality / tokenizer | `Adapter.embed / unembed` | `ADAPTERS` |
| latent space (VAE) | codec `encode / decode` | `CODECS` |
| backbone | `forward(tokens (B,N,D), c (B,D))` | `BACKBONES` |
| whole model (e.g. a UNet) | `forward(x, t, cond, drop_cond)` + `supports_guidance` | `MODELS` |
| conditioning | `forward(cond, batch_size, device, drop)` -> (B, D) | `CONDITIONERS` |
| data / prompts | `sample(n, rng)`, plus `encode(prompts, device)` for prompts | `DATASETS` / `PROMPTS` |

See `examples/custom_plugin.py`.

## RL details worth knowing

- **Log-prob reduction.** Continuous processes default to `reduction: mean` over dimensions, as DDPO and Flow-GRPO do, so per-step ratios stay near 1. Use a tiny `clip_range` (1e-4) with it. Masked text defaults to `sum` over tokens, with a PPO-style `clip_range` of about 0.2.
- **Deterministic steps are skipped.** The last DDPM step to t=0 has no noise and so no log-prob. RL trains only the stochastic steps and refuses an ODE (`eta: 0`, `noise_level: 0`) sampler.
- **CFG inside the policy.** With `guidance_scale != 1`, the guided output defines the policy at rollout and at rescoring (Flow-GRPO style).
- **KL.** It is exact per step: Gaussian mean difference over 2σ², or for text, categorical KL weighted by the unmask probability. With LoRA the reference is the same model with adapters off, so there is no second copy in memory.
- **`timestep_fraction`** trains a random subset of steps per minibatch (DanceGRPO / Flow-GRPO style).

## Toy results (CPU, ~3 min per config)

| config | reward, RL epoch 0 → 39 | notes |
|---|---|---|
| `text_masked_grpo` | 0.26 → 0.52 vowel fraction | reward-hacks into vowel spam with no KL, as expected |
| `image_ddpm_ddpo` | 0.05 → 0.995 brightness | saturates |
| `video_flow_grpo` | −0.0065 → −0.0033 | KL keeps it near the base |
| `image_flow_grpo` | 0.03 → 0.08 red dominance, noisy | the blue class drifts to red; needs tuning |

## Not built yet

See the layer map in [docs/GOAL_AND_PLAN.md](docs/GOAL_AND_PLAN.md). Within the existing packages:

- Multi-GPU (DDP/FSDP). Trainers are single-process.
- Token-level cross-attention conditioning (pooled `vector` conditioning works) and loaders for pretrained weights (SD3/FLUX/Wan/LLaDA).
- Offline preference objectives (Diffusion-DPO) and reward-gradient methods (DRaFT, ReFL).
- Confidence-based remasking for masked diffusion. It isn't a policy with log-probs, so it would be an inference-only sampler.
