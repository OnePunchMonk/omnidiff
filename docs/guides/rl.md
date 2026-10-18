# Guide: rl

Online RL on top of an SFT checkpoint. Each epoch samples prompts, rolls out a group of samples per prompt, scores them with the reward, computes group-relative advantages, and takes PPO-clipped steps with exact per-step log-probabilities.

Needs an SFT checkpoint (see [sft](sft.md)).

## Tiny run (CPU)

```bash
uv run diffusion-omni rl -c configs/image_flow_grpo.yaml device=cpu rl.init_from=runs/guide/img/sft/final rl.epochs=2 rl.prompts_per_epoch=2 rl.group_size=2 rl.rollout_batch_size=4 rl.train_batch_size=4 output_dir=runs/guide/img
uv run diffusion-omni rl -c configs/text_masked_grpo.yaml device=cpu rl.init_from=runs/guide/txt/sft/final rl.epochs=2 rl.prompts_per_epoch=2 rl.group_size=2 rl.rollout_batch_size=4 rl.train_batch_size=4 output_dir=runs/guide/txt
```

Output: `saved runs/guide/<name>/rl/final`. Per-step metrics (`reward`, `reward_std`, `pg_loss`, `ratio`, `clip_frac`, `grad_norm`) go to `runs/guide/<name>/rl/log.jsonl`.

## Full run

```bash
uv run diffusion-omni rl -c configs/image_flow_grpo.yaml
```

The shipped config runs 40 epochs, with LoRA (rank 8) on the flow model, starting from `runs/image_flow_grpo/sft/final`.

## Choosing the algorithm and reward

```yaml
rl:
  algorithm: {type: grpo, clip_range: 1e-4, kl_coef: 0.0}   # or {type: ddpo, clip_range: 1e-4}
  reward: {type: channel_dominance, channel: 0}              # or brightness, char_fraction, composite, ...
  lora: {rank: 8, alpha: 16}                                 # or null for a full fine-tune
  timestep_fraction: 0.6                                     # fraction of stochastic steps trained per minibatch
```

- `kl_coef > 0` adds an exact per-step KL to the reference. With LoRA the reference is the same model with adapters disabled.
- RL requires a stochastic sampler. `eta: 0` or `noise_level: 0` is rejected.
- A DMD2 or consistency student can be RL-tuned from its own checkpoint, by setting `rl.init_from` to it. See [distill](distill.md).

Reference: [posttrain](../reference/posttrain.md), [reward](../reference/reward.md).
