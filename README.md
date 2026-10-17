# omnidiff

RL post-training and inference for **language, image and video diffusion models**, built in plain PyTorch and NumPy. Every component plugs in through a registry.

The only dependencies are `torch`, `numpy` and `pyyaml`. It has no `diffusers`, `transformers`, `accelerate` or `trl`, so everything that runs is in this repo.

## The one idea

Every diffusion sampler is an MDP. The state is `(x_t, t, cond)`, the action is `x_next`, and the reward comes at the end. If each reverse step has an exact log-probability, the same PPO/GRPO code trains all three modalities:

| process (`process.type`) | modality | reverse step = policy | log π(x_next \| x_t) |
|---|---|---|---|
| `masked`: absorbing discrete diffusion (MDLM / LLaDA) | text | unmask each masked token w.p. (t−s)/t, draw it from the predicted categorical | exact: categorical + Bernoulli terms |
| `gaussian`: VP diffusion, ε-prediction, DDIM(η) | image, video | Gaussian, η > 0 (η = 1 ≈ DDPM, the DDPO policy) | exact Gaussian |
| `flow`: rectified flow, v-prediction | image, video | marginal-preserving reverse SDE from Flow-GRPO | exact Gaussian |

`DiffusionProcess.transition(pred, x_t, t, t_next, x_next=None)` samples a step. Pass `x_next` and it scores a stored action instead. Rollouts and RL updates go through that same call, so the ratio for a fresh rollout is exactly 1 (`tests/test_e2e.py::test_rollout_log_probs_are_on_policy`).

## Layout

```
omnidiff/
  registry.py          string -> constructor registries, context injection, `pkg.mod:Class` specs
  config.py            YAML + dotted CLI overrides
  build.py             config -> Components (codec, process, model, prompts)
  processes/           gaussian.py  flow.py  masked.py   (forward noising, training loss, policy step, step KL)
  nn/
    adapters.py        tokens | image_patch | video_patch   modality <-> (B, N, D) tokens
    backbones.py       dit: adaLN-Zero transformer, modality-agnostic
    conditioners.py    none | class | vector             (cond dict -> (B, D), null branch for CFG)
    denoiser.py        adapter -> backbone -> adapter; the model contract
    lora.py            LoRA; disabled LoRA == frozen reference policy for KL
  sampling.py          predict (with CFG), step, sample -> Trajectory
  codecs.py            pixel | text (byte tokenizer)     data <-> process tensors (VAE goes here)
  datasets.py          toy SFT data: blobs | moving_blob | text_lines
  prompts.py           none | class | text               prompt sampling + encoding to cond
  rewards.py           numpy rewards + composite
  rl/algorithms.py     PPO-clip surrogate; grpo (group advantages) | ddpo (batch advantages); KL
  rl/trainer.py        rollout -> reward -> advantages -> per-step clipped updates
  sft.py               denoising pretraining / SFT (the base policy RL starts from)
  pipeline.py          inference API
configs/               one file per (modality, process, algorithm); each holds sft + rl + sampling
examples/              custom_plugin.py
```

## Quickstart

```bash
pip install -e ".[dev]"
pytest                                   # ~1 s, CPU

# each config runs the whole loop on a laptop CPU in minutes
python -m omnidiff sft -c configs/image_flow_grpo.yaml
python -m omnidiff rl  -c configs/image_flow_grpo.yaml
python -m omnidiff sample --ckpt runs/image_flow_grpo/rl/final --prompt 0 --prompt 1 --prompt 2 --prompt 3

python -m omnidiff sft -c configs/text_masked_grpo.yaml
python -m omnidiff rl  -c configs/text_masked_grpo.yaml
python -m omnidiff sample --ckpt runs/text_masked_grpo/rl/final --prompt "the cat" --prompt "a dog"

python -m omnidiff list                  # everything registered
```

Override any key from the command line: `python -m omnidiff rl -c configs/video_flow_grpo.yaml rl.lr=1e-4 rl.lora=null device=cuda`.

From Python:

```python
from omnidiff import Pipeline
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

Every config entry is `{type: name, ...kwargs}`. A `type` can be a registered name or any import path (`my_pkg.rewards:Aesthetic`). Modules listed under `plugins:` are imported first, so their `@REGISTRY.register(...)` decorators run.

Constructors receive shared **context** by naming it in their signature: `dim`, `sample_shape`, `codec`, `mask_id`, `vocab_size`. That's how `TokenAdapter` gets the vocabulary from the tokenizer without the config repeating it.

| to add | implement | register in |
|---|---|---|
| reward model | `__call__(samples, prompts) -> np.ndarray (B,)` | `REWARDS` |
| RL objective | `advantages(rewards, groups)`, `loss(logp, old_logp, adv, kl) -> LossOutput`, `kl_coef` | `ALGORITHMS` |
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

## Not built yet

- Multi-GPU (DDP/FSDP). The trainer is single-process.
- Token-level cross-attention conditioning (pooled `vector` conditioning works today) and loaders for pretrained weights (SD3/FLUX/Wan/LLaDA). Both slot into `MODELS` / `CONDITIONERS` / `CODECS`.
- Offline preference objectives (Diffusion-DPO, D3PO) and reward-gradient methods (DRaFT, ReFL).
- Faster masked samplers (confidence-based remasking). These are deterministic selections, not policies with log-probs, so they would be inference-only.
