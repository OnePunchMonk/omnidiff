# Guide: export

Write a student as a bundle: weights in safetensors (or torch) format, plus JSON that says how to sample them. The bundle is meant to be read by other tools; `load_bundle` is the reference reader.

Needs a distilled checkpoint (see [distill](distill.md)).

```bash
uv run diffusion-omni export --ckpt runs/guide/dmd2/distill/final --out runs/guide/bundle
```

Output: `wrote runs/guide/bundle`. The bundle directory contains:

- `model.safetensors` (or `model.pt` when `safetensors` is not installed; the root dev group includes it)
- `model.json`: the model spec (adapter, backbone, conditioner)
- `sampler.json`: process, sampler, timesteps, `num_steps`, `guidance_scale`, `sample_shape`, codec, prompts, and the weights' file and sha256
- `README.md`: how to use the bundle
- `resources/`: files the spec refers to, when there are any

The checkpoint's weights are checked against the sha256 when the bundle is loaded, and a tampered file is rejected.

## Load a bundle

```python
from diffusion_omni.distill.export import load_bundle

pipe = load_bundle("runs/guide/bundle", device="cpu")
gen = pipe.generate(prompts=[0], seed=0)
print(gen.samples.shape)   # (1, 3, 16, 16) for the tiny image student
```

`load_bundle` imports the plugin modules named in `sampler.json`. Only load bundles you trust.

## Rules

- The output directory must not be the checkpoint itself or an existing diffusion-omni checkpoint.
Reference: [distill](../reference/distill.md).
