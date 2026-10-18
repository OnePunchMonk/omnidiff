# Guide: sample

Generate from a trained checkpoint. A checkpoint is a directory with `config.yaml` and `model.pt`, written by `sft`, `rl` or `distill`.

Needs a checkpoint first. The commands below use the tiny checkpoints from the [sft](sft.md) and [rl](rl.md) guides. Full-size checkpoints work the same way.

## Image or video

```bash
uv run diffusion-omni sample --ckpt runs/guide/img/rl/final --prompt 0 --prompt 1 --steps 4 --device cpu --out runs/guide/img/samples.png
```

- `--prompt` is repeatable. For class-conditional models it is a class id.
- `--steps` sets the number of sampling steps. Without it, the config's `sampling.num_steps` (or `timesteps`) is used.
- `--guidance` overrides `sampling.guidance_scale`.
- `--out` sets the PNG path. The `.npy` with the same stem holds the float samples in [0, 1]. Without `--out`, the file is `<ckpt>/samples.png`.
- `--scale` upscales the PNG (default 4). `-n` is the number of samples when no prompt is given (default 8).

Output: `wrote runs/guide/img/samples.png`.

## Text

```bash
uv run diffusion-omni sample --ckpt runs/guide/txt/rl/final --prompt "the cat" --device cpu
```

Each completion is printed as `[prompt] text`. The tiny checkpoint is trained for two RL epochs, so its text is still rough.

## From Python

```python
from diffusion_omni.core import Pipeline

pipe = Pipeline.from_checkpoint("runs/guide/txt/rl/final", device="cpu")
gen = pipe.generate(prompts=["the sun"], num_steps=16, seed=0)
print(gen.samples)
```

`Pipeline.from_checkpoint(path, device, overrides)` accepts dotted overrides, for example `["sampling.guidance_scale=2.0"]`.

Reference: [core](../reference/core.md).
