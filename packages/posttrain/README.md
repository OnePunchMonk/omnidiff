# diffusion-omni-posttrain

Supervised denoising training and online RL for text, image and video diffusion.

- `diffusion-omni sft`: denoising loss on a dataset. Produces the base policy.
- `diffusion-omni rl`: rollout, score, advantages, PPO-clipped updates with exact per-step KL. Algorithms: `grpo` and `ddpo` (`ALGORITHMS` registry). LoRA or full fine-tune.

```bash
uv run diffusion-omni sft -c configs/image_flow_grpo.yaml
uv run diffusion-omni rl  -c configs/image_flow_grpo.yaml
```

Reference: [docs/reference/posttrain.md](../../docs/reference/posttrain.md). Guides: [sft](../../docs/guides/sft.md), [rl](../../docs/guides/rl.md).
