# diffusion-omni-core

Shared foundation: diffusion processes (masked, Gaussian, flow), modality adapters, the DiT backbone, CFG, LoRA, sampling, the `Pipeline` inference API, and the `diffusion-omni` CLI (`sample`, `list`). Depends only on torch, numpy and pyyaml.

```bash
uv run diffusion-omni list
uv run diffusion-omni sample --ckpt runs/<name>/<stage>/final --prompt 0 --steps 4
```

Reference: [docs/reference/core.md](../../docs/reference/core.md).
