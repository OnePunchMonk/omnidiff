# diffusion-omni

A diffusion monorepo for text, image and video, written in PyTorch and NumPy. It has a shared core, SFT and RL post-training, few-step distillation, and a scorecard that compares students with their teacher across step counts. Components are built from YAML configs through registries, and packages plug in through entry points.

`diffusion-omni-core` depends only on `torch`, `numpy` and `pyyaml`.

## Status

| package | what is on `main` | status |
|---|---|---|
| [core](packages/core/README.md) | processes (masked, Gaussian, flow), DiT, adapters, CFG, LoRA, sampling, `Pipeline`, `diffusion-omni sample`, `diffusion-omni list` | v0.1 |
| [posttrain](packages/posttrain/README.md) | `diffusion-omni sft`, `diffusion-omni rl` (GRPO, DDPO) | v0.1 |
| [distill](packages/distill/README.md) | `diffusion-omni distill`, `diffusion-omni export`; 10 methods (kd, dmd2, dmd2_reward, consistency, flow_map, rcm, progressive, sdtt, coupled_kd, dcd) | v0 |
| [reward](packages/reward/README.md) | toy rewards, set metrics, `diffusion-omni scorecard` | seed |
| [data](packages/data/README.md) | three synthetic datasets | seed |

Reference pages for each package are in [docs/reference/](docs/reference/). Task guides are in [docs/guides/](docs/guides/).

### In review (not on `main`)

These exist only on unmerged branches. They are not available in this repo as it stands, and no command or config below depends on them:

- reward service (a batched reward/eval server)
- miles-diffusion adapter
- eval package (shared eval harness)
- reliability work on rewards (agreement and hacking probes)

## The idea

Every diffusion sampler is a Markov decision process. The state is `(x_t, t, cond)`, the action is `x_next`, and the reward arrives at the end. When each reverse step has an exact log-probability, the same PPO/GRPO code trains all three modalities.

| process (`process.type`) | modality | reverse step (the policy) | log π(x_next \| x_t) |
|---|---|---|---|
| `masked`: absorbing discrete diffusion | text | unmask each masked token with probability (t−s)/t, drawn from the predicted categorical | exact: categorical and Bernoulli terms |
| `gaussian`: VP diffusion, ε-prediction, DDIM(η) | image, video | Gaussian when η > 0 (η = 1 is DDPM, the DDPO policy) | exact Gaussian |
| `flow`: rectified flow, v-prediction | image, video | marginal-preserving reverse SDE (Flow-GRPO) | exact Gaussian |

`DiffusionProcess.transition(pred, x_t, t, t_next, x_next=None)` samples a step. If you pass `x_next`, it scores that stored action instead. Rollouts and RL updates use the same call, so the ratio on a fresh rollout is exactly 1 (`packages/posttrain/tests/test_posttrain_e2e.py::test_rollout_log_probs_are_on_policy`).

## Layout

```
packages/
  core/       diffusion-omni-core       diffusion_omni.core       processes, nn, sampling, pipeline, CLI (torch, numpy, pyyaml)
  posttrain/  diffusion-omni-posttrain  diffusion_omni.posttrain  SFT, GRPO, DDPO
  distill/    diffusion-omni-distill    diffusion_omni.distill    few-step distillation, export
  reward/     diffusion-omni-reward     diffusion_omni.reward     rewards, set metrics, scorecard
  data/       diffusion-omni-data       diffusion_omni.data       synthetic datasets
configs/      one YAML per recipe; configs/distill/ has one per distillation run
docs/         architecture, goal and plan, reference, guides, scope, results, notes
examples/     custom_plugin.py: registering your own components
scripts/      toy report and diagnostic scripts; check_docs.py
```

Each package is a separate distribution under the `diffusion-omni.*` namespace. A package plugs in through two entry-point groups: `diffusion_omni.plugins` imports its components so their registrations run, and `diffusion_omni.commands` adds CLI subcommands. Installing a package is enough to make its components usable from config.

Inside `diffusion_omni.core`:

```
registry.py      name -> constructor registries, shared context injection, "pkg.mod:Class" specs, entry-point discovery
config.py        YAML loading and dotted command-line overrides
build.py         config -> Components (codec, process, model, prompts, sampling)
processes/       gaussian.py, flow.py, masked.py (forward noising, training loss, policy step, step KL)
nn/              adapters (tokens, image_patch, video_patch), DiT backbone, conditioners, denoiser, LoRA
sampling.py      predict (with CFG), step, sample -> Trajectory
codecs.py        pixel, text (byte tokenizer)
prompts.py       none, class, text
pipeline.py      inference API (Pipeline)
testing.py       tiny_config(): shrink any config to seconds on CPU
cli.py           diffusion-omni sample, diffusion-omni list
```

## Quickstart

Setup and tests:

```bash
uv sync                    # installs every workspace package in editable mode into .venv
uv run pytest              # 165 tests, about 10 seconds on CPU
uv run diffusion-omni list       # every component registered by the installed packages
uv run diffusion-omni --help     # all subcommands
```

Full-size shipped recipes. These are the paths the configs write to. They have not been timed here; the tiny versions in [docs/guides/](docs/guides/) are the ones checked by the docs test.

```bash
# image: flow matching, then Flow-GRPO with LoRA
uv run diffusion-omni sft -c configs/image_flow_grpo.yaml
uv run diffusion-omni rl  -c configs/image_flow_grpo.yaml
uv run diffusion-omni sample --ckpt runs/image_flow_grpo/rl/final --prompt 0 --prompt 1 --prompt 2 --prompt 3

# text: masked diffusion, then GRPO
uv run diffusion-omni sft -c configs/text_masked_grpo.yaml
uv run diffusion-omni rl  -c configs/text_masked_grpo.yaml
uv run diffusion-omni sample --ckpt runs/text_masked_grpo/rl/final --prompt "the cat" --prompt "a dog"

# few-step distillation of the image teacher, then compare step counts
uv run diffusion-omni distill -c configs/distill/image_flow_dmd2.yaml
uv run diffusion-omni scorecard --ref runs/image_flow_grpo/sft/final \
    --ckpt dmd2=runs/distill/image_flow_dmd2/distill/final --nfe 1 2 4 8 --out runs/scorecard
```

Override any key from the command line with `key=value`: `uv run diffusion-omni rl -c configs/video_flow_grpo.yaml rl.lr=1e-4 rl.lora=null device=cuda`.

From Python:

```python
from diffusion_omni.core import Pipeline

pipe = Pipeline.from_checkpoint("runs/text_masked_grpo/rl/final")
gen = pipe.generate(prompts=["the sun"], num_steps=32, seed=0)
print(gen.samples)   # list[str] for text, float numpy array in [0, 1] for pixels
```

Guides with tiny CPU versions of each step: [sample](docs/guides/sample.md), [sft](docs/guides/sft.md), [rl](docs/guides/rl.md), [distill](docs/guides/distill.md), [export](docs/guides/export.md), [eval-scorecard](docs/guides/eval-scorecard.md).

## Shipped configs

| config | modality | process | algorithm | reward |
|---|---|---|---|---|
| `text_masked_grpo` | text, 32 bytes | masked | GRPO, full fine-tune | fraction of vowels in the completion |
| `image_flow_grpo` | image 3×16×16, class-conditional | flow | GRPO, LoRA | channel dominance (red) |
| `image_flow_grpo_ts10` | as `image_flow_grpo`, with `model.time_scale: 10` (smooth in t, for flow-map students) | flow | GRPO, LoRA | channel dominance (red) |
| `image_ddpm_ddpo` | image 3×16×16, class-conditional | gaussian, η = 1 | DDPO, full fine-tune | brightness |
| `video_flow_grpo` | video 3×4×16×16, class-conditional | flow | GRPO, LoRA, KL 0.01 | brightness + temporal smoothness |
| `video_flow_grpo_ts10` | as `video_flow_grpo`, with `model.time_scale: 10` | flow | GRPO, LoRA | brightness + temporal smoothness |

Distillation recipes are in `configs/distill/` (image, video and text, one file per method and modality). Their `teacher:` key points to an SFT checkpoint.

The data and rewards are synthetic. They exist so the loop can be run and checked, and they are not benchmarks.

## Plugging things in

Every config entry is `{type: name, ...kwargs}`. `type` can be a registered name or an import path such as `my_pkg.rewards:Aesthetic`. Modules listed under `plugins:` are imported before the build, so their `@REGISTRY.register(...)` decorators run. Installed packages that declare an `diffusion_omni.plugins` entry point load automatically.

Constructors get shared context by naming it in their signature: `dim`, `sample_shape`, `codec`, `mask_id`, `vocab_size`.

| to add | implement | register in |
|---|---|---|
| reward | `__call__(samples, prompts) -> np.ndarray (B,)` | `REWARDS` |
| RL objective | `advantages(rewards, groups)`, `loss(...) -> LossOutput`, `kl_coef` | `ALGORITHMS` |
| diffusion process | `DiffusionProcess` subclass | `PROCESSES` |
| modality adapter | `embed`, `unembed` | `ADAPTERS` |
| latent space (VAE) | codec `encode`, `decode` | `CODECS` |
| backbone | `forward(tokens (B, N, D), c (B, D))` | `BACKBONES` |
| whole model | `forward(x, t, cond, drop_cond)`, `supports_guidance` | `MODELS` |
| conditioner | `forward(cond, batch_size, device, drop) -> (B, D)` | `CONDITIONERS` |
| dataset | `sample(n, rng) -> (x0, cond)` | `DATASETS` |
| prompt source | `sample(n, rng)`, `encode(prompts, device)` | `PROMPTS` |
| distillation method | `Distiller` subclass | `DISTILLERS` |

`examples/custom_plugin.py` shows a reward, a conditioner, and loading a plugin from a config. Per-package extension points are in [docs/reference/](docs/reference/).

## Results

Toy results are in [docs/results/](docs/results/). The main file is [distillation_toy.md](docs/results/distillation_toy.md), which covers the distillation methods, the order study (RL and distillation), and the flow-map diagnosis. [g2_kl_sweep.md](docs/results/g2_kl_sweep.md) and [g2_order_study_table.md](docs/results/g2_order_study_table.md) hold the order-study tables. All results are CPU, single-seed or small-seed toy runs.

## Not built yet

- Multi-GPU training (DDP or FSDP). Trainers are single-process.
- Token-level cross-attention conditioning. Pooled `vector` conditioning works. Loaders for pretrained weights (SD3, FLUX, Wan, LLaDA) do not exist.
- Offline preference objectives (Diffusion-DPO) and reward-gradient methods beyond `dmd2_reward` (DRaFT, ReFL).
- Confidence-based remasking for masked diffusion. It has no log-probability policy, so it would be an inference-only sampler.
- The data engine (recaptioning, filtering, deduplication, latent precompute).
- A batched reward and eval harness on `main`. See "In review" above.

The layer map and roadmap are in [docs/GOAL_AND_PLAN.md](docs/GOAL_AND_PLAN.md). Distillation scope is in [docs/scope/distillation.md](docs/scope/distillation.md). The architecture is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
