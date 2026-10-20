# Reference: diffusion-omni-core

Package: `packages/core`, import name `diffusion_omni.core`. Dependencies: torch, numpy, pyyaml.

## Purpose

Processes (the forward and reverse diffusion steps, each with an exact log-probability where one exists), modality adapters, a DiT backbone, conditioners, CFG, LoRA, sampling, the `Pipeline` inference API, and the `diffusion-omni` CLI entry point (`sample`, `list`).

## Public API

From `diffusion_omni.core`: `Pipeline`, `Generation`, `Components`, `build`, `load_config`, `Trajectory`, `predict`, `sample`, `step`, `reference_sample`, `parity`, `Parity`.

| module | names |
|---|---|
| `pipeline.py` | `Pipeline.from_config(cfg)`, `Pipeline.from_checkpoint(path, device=None, overrides=())`, `Pipeline.generate(prompts, n, num_steps, guidance_scale, seed, return_trajectory, timesteps)`, `Pipeline.reference(prompts, n, num_steps, guidance_scale, seed, dtype)`; `Generation(samples, prompts, raw, trajectory)` |
| `build.py` | `build(cfg, device=None) -> Components`; `Components(cfg, device, sample_shape, codec, process, model, prompts, sampling, context)` |
| `config.py` | `load_config(path, overrides)`, `apply_overrides(cfg, overrides)`, `parse_value`, `save_config`, `from_dict`, `SamplingConfig` |
| `sampling.py` | `sample(model, process, shape, cond, num_steps, guidance_scale, generator, keep_trajectory, timesteps)`, `step(...)`, `predict(...)`, `resolve_timesteps(...)`, `Trajectory` |
| `reference.py` | `reference_sample(model, process, shape, cond, num_steps=500, guidance_scale, generator, device, dtype=float64, x_init, timesteps)`: probability-flow ODE with Heun (gaussian, flow) or ancestral (masked) at high NFE, ignoring the process's `sampler`; `parity(candidate, reference, atol, min_match) -> Parity(max_abs, rmse, match, passed)` |
| `processes/base.py` | `DiffusionProcess` (abstract), `ContinuousProcess`, `Transition`, `reduce`, `gaussian_log_prob` |
| `processes/flow.py` | `FlowMatching` (`process.type: flow`) |
| `processes/gaussian.py` | `GaussianDiffusion` (`process.type: gaussian`) |
| `processes/masked.py` | `MaskedDiffusion` (`process.type: masked`) |
| `nn/adapters.py` | `TokenAdapter`, `ImagePatchAdapter`, `VideoPatchAdapter` |
| `nn/backbones.py` | `DiT` |
| `nn/conditioners.py` | `NullConditioner`, `ClassConditioner`, `VectorConditioner` |
| `nn/denoiser.py` | `Denoiser`, `build_model(spec, **context)` |
| `nn/lora.py` | `LoRALinear`, `apply_lora(model, rank, alpha, dropout, targets)`, `has_lora`, `lora_disabled`, `merged_state_dict` |
| `codecs.py` | `PixelCodec`, `ByteTokenizer`, `TextCodec` |
| `prompts.py` | `Unconditional`, `ClassPrompts`, `TextPrompts` |
| `utils.py` | `seed_everything`, `resolve_device`, `make_generator`, `save_checkpoint`, `load_weights`, `JsonlLogger`, `to_grid`, `write_png` |
| `testing.py` | `tiny_config(path, output_dir, *extra)` |

## CLI

Output of `uv run diffusion-omni --help` (abridged to core's own commands, the others come from installed packages):

```
usage: diffusion-omni [-h] {sample,list,distill,export,scorecard,sft,rl} ...
```

### `diffusion-omni sample`

```
usage: diffusion-omni sample [-h] --ckpt CKPT [-n N] [--prompt PROMPT]
                       [--steps STEPS] [--guidance GUIDANCE] [--seed SEED]
                       [--device DEVICE] [--out OUT] [--scale SCALE]
                       [overrides ...]
```

Pixel and video models write a PNG grid (default `<ckpt>/samples.png`) and a `.npy` with the same stem. Text models print each sample. `--prompt` is repeatable: a class id for class-conditional models, a text prefix for text models.

### `diffusion-omni list`

Prints every registry and its registered names, after loading installed plugins.

## Config keys

Top level (read by `build`):

| key | meaning |
|---|---|
| `seed`, `device`, `output_dir` | run settings; `device` is `auto` (CUDA if available, else CPU), `cpu`, `cuda`, or any torch device string |
| `sample_shape` | per-sample shape, e.g. `[3, 16, 16]`, `[3, 4, 16, 16]` (video), `[32]` (text) |
| `codec` | `pixel` or `text`; a string or `{type: ...}` |
| `process` | `{type: flow|gaussian|masked, ...}` (see below) |
| `model` | `{dim, adapter, backbone, conditioner}`; `type` defaults to `denoiser` |
| `prompts` | `none`, `class`, or `text` (plus its kwargs) |
| `sampling` | `num_steps` (default 20), `guidance_scale` (default 1.0), `timesteps` (explicit grid, overrides `num_steps`) |
| `plugins` | list of modules to import before building |

Process keys:

| process | keys (default) |
|---|---|
| `flow` | `noise_level` (0.7), `shift` (1.0), `time_sampling` (`logit_normal` or `uniform`), `reduction` (`mean`), `sampler` (`sde`, `ode`, `renoise`) |
| `gaussian` | `schedule` (`vp_linear` or `cosine`), `eta` (1.0), `clip_x0`, `reduction` (`mean`), `beta_min`, `beta_max`, `sampler` (`ddim`, `renoise`, `ode`) |
| `masked` | `temperature` (1.0), `reduction` (`sum`), `sampler` (`ancestral`, `renoise`, `coupled`), `coupled_noise` (`gumbel`) |

Model keys: `dim`; `adapter` (`tokens`, `image_patch` with `patch`, `video_patch` with `patch` and `temporal_patch`); `backbone` (`dit` with `depth`=4, `heads`=4, `mlp_ratio`); `conditioner` (`none`, `class` with `num_classes`, `vector` with `in_dim`).

## Registries

| registry | registered names |
|---|---|
| `PROCESSES` | `flow`, `gaussian`, `masked` |
| `ADAPTERS` | `tokens`, `image_patch`, `video_patch` |
| `BACKBONES` | `dit` |
| `CONDITIONERS` | `none`, `class`, `vector` |
| `MODELS` | `denoiser` |
| `CODECS` | `pixel`, `text` |
| `PROMPTS` | `none`, `class`, `text` |

Other registries live in other packages: `DATASETS`, `REWARDS` (reward), `ALGORITHMS` (posttrain), `DISTILLERS` (distill). `METRICS` is declared in `registry.py` and has no registered entries yet.

## Extension points

Implement the class, then register it, or refer to it as `type: my_pkg.mod:MyClass` without registering.

- **Process**: subclass `DiffusionProcess` (or `ContinuousProcess` for a schedule with `alpha_sigma`). Implement `timesteps`, `prior`, `training_loss`, `transition(pred, x_t, t, t_next, x_next=None)`, and `kl`. Register in `PROCESSES`.
- **Adapter**: `embed` and `unembed`. Register in `ADAPTERS`.
- **Backbone**: `forward(tokens (B, N, D), c (B, D))`. Register in `BACKBONES`.
- **Whole model**: `forward(x, t, cond, drop_cond)` and a `supports_guidance` attribute. Register in `MODELS`.
- **Conditioner**: `forward(cond, batch_size, device, drop) -> (B, D)`. Register in `CONDITIONERS`.
- **Codec** (a VAE goes here): `encode`, `decode`. Register in `CODECS`.
- **Prompt source**: `sample(n, rng)` and `encode(prompts, device)`. Register in `PROMPTS`.

Constructors receive context values (`dim`, `sample_shape`, `codec`, `mask_id`, `vocab_size`) when they name them in their signature.

Example: [examples/custom_plugin.py](../../examples/custom_plugin.py).

Tests: `packages/core/tests/test_core_components.py`, `packages/core/tests/test_processes.py`, `packages/core/tests/test_distill_contract.py`.
