# 2. From SFT to RL

Goal: understand the two training stages in `diffusion-omni-posttrain`, run both on a tiny image model and a tiny text model, and read the RL log.

Prerequisites: the setup block in [README.md](README.md) and the image teacher from [tutorial 1](01-denoising-as-a-policy.md), Step 1 (`$OUT/img/sft/final`).

## The idea in plain words

Stage 1, SFT, is ordinary denoising training: take data, corrupt it, train the network to undo the corruption. This gives a teacher that samples from the data distribution but does not know about any reward.

Stage 2, RL, treats sampling as the policy from tutorial 1. It samples several images per prompt, scores each finished image with a reward, and nudges the network so that the steps that led to high-reward images become more likely. The algorithm is GRPO: instead of a learned value function, each sample's advantage is its reward relative to the other samples of the same prompt.

## The math, as implemented

**SFT** (`packages/posttrain/src/diffusion_omni/posttrain/sft.py`) calls `process.training_loss(model, x0, cond, drop_cond)`:

- Flow: sample `t` (logit-normal by default), `x_t = (1 - t) x0 + t * noise`, loss `MSE(model(x_t, t), noise - x0)`.
- Masked: sample `t`, mask each free token with probability `t`, loss is the cross-entropy on masked positions weighted by `1 / t` (the MDLM weight for the linear schedule), summed and divided by the number of maskable positions.
- `sft.cond_dropout` randomly replaces the condition by its null value, so the same model can later be sampled with classifier-free guidance.

**RL** (`packages/posttrain/src/diffusion_omni/posttrain/rl/trainer.py`, `rl/algorithms.py`). One epoch is:

1. Sample `prompts_per_epoch` prompts, repeat each `group_size` times, and roll out one trajectory each with the current model (`keep_trajectory=True`).
2. Decode the final samples and score them: `rewards = reward(decoded_samples, prompts)`.
3. Advantages. For `grpo`, `adv = (r - mean_group(r)) / (std_group(r) + 1e-4)`, clipped to `[-5, 5]`. For `ddpo` the mean and std are taken over the whole batch (`adv_mode: batch`). Every step of a trajectory shares its sample's advantage.
4. For each minibatch and for a random `timestep_fraction` of the stochastic steps, re-score the stored `x_next` with the current model (tutorial 1), and compute the PPO-clipped loss

```text
ratio = exp(logp_new - logp_old)
loss  = mean( max( -adv * ratio, -adv * clip(ratio, 1 - clip_range, 1 + clip_range) ) )
        + kl_coef * mean( KL(new step, reference step) )
```

5. One AdamW step per minibatch, gradient norm clipped to `max_grad_norm`.

Details that matter:

- **The KL term is exact and per step.** It compares the Gaussian (or categorical) step distributions of the current model and a reference at the same state, via `process.kl`. With `rl.lora` set, the reference is the same network with LoRA switched off, so the reference costs no extra memory. Without LoRA a frozen deep copy is made.
- **LoRA.** `rl.lora: {rank, alpha}` trains low-rank adapters only. The saved checkpoint has them merged into the weights (`merged_state_dict`), so an RL checkpoint loads like any other.
- **Only stochastic steps are trained.** Steps whose `log_prob` is `None` are skipped (tutorial 1).

## Walkthrough

### Step 1. RL on the image teacher

The reward is `channel_dominance` on channel 0: mean of the red channel minus the mean of the others, i.e. "make it red" (`packages/reward/src/diffusion_omni/reward/rewards.py`).

```bash
uv run diffusion-omni rl -c configs/image_flow_grpo.yaml "${TINY_IMG[@]}" \
  rl.init_from="$OUT/img/sft/final" rl.epochs=30 rl.prompts_per_epoch=4 rl.group_size=8 \
  rl.rollout_batch_size=32 rl.train_batch_size=16 rl.lr=1e-3 output_dir="$OUT/img"
```

The overrides shrink the config: 30 epochs of 4 prompts x 8 samples instead of 40 epochs of 8 x 8, and `rl.lr=1e-3` instead of `3e-4`, because a tiny LoRA needs a larger step to move in 30 epochs. It runs in about two seconds.

Expected on the reference run (`reward` in the log, step 0 to step 29):

```text
step=0  reward=0.0156 reward_std=0.0307 ... clip_frac=0.48
step=15 reward=0.1748 ...
step=29 reward=0.2527 reward_std=0.0373 ...
```

The mean reward rises from about 0.01 to about 0.25. The full log is `$OUT/img/rl/log.jsonl`. Columns:

- `reward`, `reward_std`: mean and spread of the rollout rewards in that epoch.
- `ratio`: mean of `exp(logp_new - logp_old)` in the update. It stays near 1.
- `clip_frac`: share of ratios outside `[1 - clip_range, 1 + clip_range]`. The shipped image configs use `clip_range: 1e-4`, so this is large even though training is nearly on-policy. We saw 0.1 to 0.5 here and did not investigate it further. `test_rollout_log_probs_are_on_policy` checks that re-scoring a fresh rollout gives the rollout's log-probs.
- `grad_norm`: before clipping.
- `kl`: appears only when `kl_coef > 0`.

Look at the samples before and after:

```bash
uv run diffusion-omni sample --ckpt "$OUT/img/sft/final" --prompt 1 --prompt 2 -n 4 --out "$OUT/img_sft.png"
uv run diffusion-omni sample --ckpt "$OUT/img/rl/final" --prompt 1 --prompt 2 -n 4 --out "$OUT/img_rl.png"
```

Each writes a PNG grid and a `.npy` next to it. After RL the samples are redder; the toy images are 16x16 blobs, so the grid is small.

### Step 2. SFT and RL on text

The text task completes a prompt ("the cat") as a byte-level masked diffusion LM. The reward is `char_fraction` with `chars: "aeiou"`: the share of completion characters that are vowels.

```bash
uv run diffusion-omni sft -c configs/text_masked_grpo.yaml "${TINY_TXT[@]}" \
  sft.steps=2000 sft.batch_size=32 sft.log_every=1000 output_dir="$OUT/txt"

uv run diffusion-omni rl -c configs/text_masked_grpo.yaml "${TINY_TXT[@]}" \
  rl.init_from="$OUT/txt/sft/final" rl.epochs=30 rl.prompts_per_epoch=6 rl.group_size=8 \
  rl.rollout_batch_size=48 rl.train_batch_size=24 rl.lr=3e-4 output_dir="$OUT/txt"

uv run diffusion-omni sample --ckpt "$OUT/txt/sft/final" --prompt "the cat" --prompt "a dog"
uv run diffusion-omni sample --ckpt "$OUT/txt/rl/final" --prompt "the cat" --prompt "a dog"
```

On the reference run the SFT teacher writes sentences from the toy corpus ("the cat sat on t e mat."), and the RL model writes strings such as `the cataoiiioiooiiauooo`. The mean reward goes from about 0.28 to 1.0 within 30 epochs. The reward was met, and the text is useless. That is reward hacking, in its simplest form, and it is the subject of [tutorial 5](05-rewards-reliability-hacking.md).

Look at the last log lines: once every sample in a group has reward 1, `reward_std` is 0, the group advantages are all exactly 0, and `grad_norm` is 0. GRPO has no signal left.

```bash
tail -n 3 "$OUT/txt/rl/log.jsonl"
```

## Failure modes we actually hit

- **Reward hacking.** Optimising a reward alone moves the model off the data distribution. In the tiny text run above, and in the image runs of [tutorial 5](05-rewards-reliability-hacking.md), the reward goes up while sample quality collapses. At the project's toy scale this is measured in [docs/results/distillation_toy.md](../results/distillation_toy.md#order-study-rl-and-distillation-image-flow-teacher-redness-reward): RL on the 4-step student with no KL reaches a higher reward than RL on the teacher and loses precision (0.32 against 0.80 for the teacher arm). The KL anchor is the fix measured in the same document.
- **Deterministic sampler.** Raises at the first rollout (tutorial 1).
- **Flat groups give no gradient.** If a group's rewards are identical, its advantages are 0.
- **A noisy on-policy ratio with a tiny `clip_range`.** See `clip_frac` above. Treat `clip_frac` as a diagnostic, not as a measure of how off-policy the update is.

## Exercises

1. Compute the GRPO advantages for rewards `[0.1, 0.3, 0.2, 0.6, 0.5, 0.4]` with groups `[0, 0, 0, 1, 1, 1]`, then the batch-normalised version (`ddpo`).
2. With `clip_range = 0.2`, advantages `[+1, -1]` and old log-probs `[0, 0]`, what is the loss and the `clip_frac` for new log-probs `[0.5, -0.5]`? And for `[-0.5, 0.5]`?
3. Why does the KL reference cost no memory when `rl.lora` is set?

Run this to check 1 and 2:

```bash
uv run python - <<'EOF'
import numpy as np, torch
from diffusion_omni.posttrain.rl.algorithms import compute_advantages, GRPO

r = np.array([0.1, 0.3, 0.2, 0.6, 0.5, 0.4])
g = np.array([0, 0, 0, 1, 1, 1])
print("group:", compute_advantages(r, g, "group").round(3))
print("batch:", compute_advantages(r, g, "batch").round(3))
print("flat :", compute_advantages(np.array([0.5] * 4), np.zeros(4, int), "group"))

algo = GRPO(clip_range=0.2)
adv, old = torch.tensor([1.0, -1.0]), torch.zeros(2)
for new in (torch.tensor([0.0, 0.0]), torch.tensor([0.5, -0.5]), torch.tensor([-0.5, 0.5])):
    out = algo.loss(new, old, adv)
    print(new.tolist(), "loss", round(out.loss.item(), 4), "clip_frac", out.metrics["clip_frac"])
EOF
```

Answers:

1. Group: `[-1.223, 1.223, 0, 1.223, 0, -1.223]`. Each group is z-scored on its own (population std), so both groups use the same set of values `{-1.223, 0, 1.223}` even though group 1's rewards are higher. Batch: `[-1.463, -0.293, -0.878, 1.463, 0.878, 0.293]`, where group 1 is above group 0 because its rewards are higher overall. The batch version keeps the prompt's reward level in the advantage. The group version removes it.
2. `[0.5, -0.5]` gives loss `-0.2` and `clip_frac` 1.0: both ratios (1.65 and 0.61) are outside `[0.8, 1.2]` in the direction the advantage wanted, so the clipped surrogate caps the gain. `[-0.5, 0.5]` gives loss `0.521` and `clip_frac` 1.0: both moved the wrong way, and `max(...)` keeps the unclipped, worse value, so the loss is not capped on that side. The first line (`[0, 0]`) is loss 0.
3. `lora_disabled(model)` (in `core/nn/lora.py`) temporarily turns the adapters off, so the same weights act as the reference.

Next: [3. Few-step distillation](03-few-step-distillation.md).
