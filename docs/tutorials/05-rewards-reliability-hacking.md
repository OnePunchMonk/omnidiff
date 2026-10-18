# 5. Rewards, reliability and hacking

Goal: write and plug in a reward, see reward hacking happen in two settings (RL and reward-weighted distillation), see what a KL anchor does, and learn how much to trust a single number.

Prerequisites: the setup block in [README.md](README.md) and the image teacher from [tutorial 1](01-denoising-as-a-policy.md) (`$OUT/img/sft/final`). Tutorial 2 explains the RL loop.

Not covered: the batched reward service, the miles adapter and the `eval` command are coming and are not merged.

## The idea in plain words

A reward is a function of decoded samples and prompts. Optimising it moves the model toward whatever the function measures, including its loopholes. The defence is to never read the reward alone: read it next to metrics that say whether samples still look like the data (precision, recall, class accuracy), and keep the model anchored to its starting point.

## The reward API

`reward(samples, prompts) -> np.ndarray` of shape `(B,)`. `samples` are decoded NumPy arrays in `[0, 1]` for image and video, or a list of `str` for text (`packages/reward/src/diffusion_omni/reward/rewards.py`). Built-ins: `channel_dominance`, `brightness`, `temporal_smoothness`, `char_fraction`, `contains`, `composite` (a weighted sum). A reward with `differentiable = True` also defines `torch_call` on a tensor in `[0, 1]`, which `dmd2_reward` uses.

Registering your own is a plugin (`examples/custom_plugin.py` registers `centered_blob`). Context such as `sample_shape` is injected when the constructor asks for it.

## Walkthrough

### Step 1. Call a reward directly

```bash
uv run python - <<'EOF'
import numpy as np
from diffusion_omni.core.registry import REWARDS
import diffusion_omni.reward  # registers the built-in rewards

red = REWARDS.build({"type": "channel_dominance", "channel": 0})
x = np.zeros((2, 3, 4, 4)); x[0, 0] = 1.0; x[1, 2] = 1.0   # a red image, a blue image
print(red(x, [None, None]))
EOF
```

Output: `[ 1. -0.5]`. Red scores 1 (channel mean 1 minus the mean of the other two, 0), blue scores `0 - 0.5`.

### Step 2. RL without and with a KL anchor

Same RL as tutorial 2, but 60 epochs, with `kl_coef` 0 and 0.3. Both use LoRA, so the reference for the KL is the teacher.

```bash
for kl in 0 0.3; do
  uv run diffusion-omni rl -c configs/image_flow_grpo.yaml "${TINY_IMG[@]}" \
    rl.init_from="$OUT/img/sft/final" rl.epochs=60 rl.prompts_per_epoch=4 rl.group_size=8 \
    rl.rollout_batch_size=32 rl.train_batch_size=16 rl.lr=1e-3 rl.algorithm.kl_coef=$kl \
    output_dir="$OUT/img_kl$kl" 2>&1 | grep -E "^step=(0|59) " | cut -c1-120
done

uv run diffusion-omni scorecard --ref "$OUT/img/sft/final" \
  --ckpt rl_kl0="$OUT/img_kl0/rl/final" --ckpt rl_kl0.3="$OUT/img_kl0.3/rl/final" \
  --nfe 4 -n 64 --ref-steps 8 --out "$OUT/sc_rl"
```

On the reference run the mean training reward after 60 epochs was 0.62 without the KL and 0.05 with it. In the scorecard (columns `precision`, `recall`, `reward`):

```text
reference      precision 0.64  recall 0.77  reward 0.026
rl_kl0         precision 0.00  recall 0.00  reward 0.631
rl_kl0.3       precision 0.31  recall 0.84  reward 0.056
```

Without the anchor the reward is about 25 times the teacher's and the samples have left the teacher's distribution entirely (precision and recall both 0, `fd` about 20). With the anchor the reward moves much less and the samples stay near the teacher. The tiny teacher is poor to begin with, so absolute precision is low; read the pattern, not the values.

At the project's toy scale ([docs/results/distillation_toy.md](../results/distillation_toy.md), "With a KL anchor"): for RL on the 4-step student, precision goes from 0.42 at `kl_coef` 0 to 0.82 at 0.1 while the reward stays about twice the teacher's. That table also has three seeds and standard deviations, and says B's KL coefficient is the more stable knob. Full table: [g2_kl_sweep.md](../results/g2_kl_sweep.md).

### Step 3. Reward hacking inside distillation

`dmd2_reward` adds the reward to DMD2's generator loss: `loss = loss_dmd2 - reward_weight * mean r(x0)` in `mode: gradient` (`methods/dmd2_reward.py`). The two gradients are on different scales; the logged `reward_grad_ratio` shows their ratio. Sweep the weight:

```bash
DIST=(device=cpu teacher="$OUT/img/sft/final" distill.pool_size=256 distill.teacher_steps=8
      distill.batch_size=16 distill.eval_every=100000 distill.log_every=300 distill.steps=300)
for w in 0 1 4; do
  uv run diffusion-omni distill -c configs/distill/image_flow_dmd2_reward.yaml "${DIST[@]}" \
    distill.method.reward_weight=$w output_dir="$OUT/dr_w$w" 2>&1 | grep -E "^step=300" | cut -c1-40
done
uv run diffusion-omni scorecard --ref "$OUT/img/sft/final" --ckpt w0="$OUT/dr_w0/distill/final" \
  --ckpt w1="$OUT/dr_w1/distill/final" --ckpt w4="$OUT/dr_w4/distill/final" \
  --nfe 4 -n 128 --ref-steps 16
```

On the reference run: reward -0.012 at weight 0, 0.61 at weight 1 and 0.95 at weight 4, with precision and recall both 0 at weights 1 and 4, i.e. the student produces saturated images the teacher never would. The same pattern, at the full toy size and with a weight where the reward helps without harm, is in [docs/notes/D2.md](../notes/D2.md) (one seed, weight 0.3 versus 1). Reward weight is a threshold, not a dial.

### Step 4. A custom reward

```bash
PYTHONPATH=. uv run diffusion-omni rl -c configs/image_flow_grpo.yaml "${TINY_IMG[@]}" \
  rl.init_from="$OUT/img/sft/final" "plugins=[examples.custom_plugin]" "rl.reward={type: centered_blob}" \
  rl.epochs=3 rl.prompts_per_epoch=2 rl.group_size=4 rl.rollout_batch_size=8 rl.train_batch_size=4 \
  output_dir="$OUT/plug" 2>&1 | tail -n 2 | cut -c1-100
```

`PYTHONPATH=.` is needed because `examples` is imported by name from the repo root.

### Step 5. How much to trust one number

The scorecard output of one run is one training seed and one set of n samples. Two tools exist:

- Train several seeds and aggregate with `diffusion_omni.reward.aggregate` (mean and sample std, `n - 1`). A cell fed by fewer values than planned is marked `(k/N)`.
- Raise `-n`. The reported std is training-seed variance only if the evaluation seed is fixed.

```bash
uv run python - <<'EOF'
from diffusion_omni.reward.aggregate import aggregate, to_markdown

rows = [{"model": "m", "nfe": 4, "seed": s, "fd": v, "recall": r}
        for s, v, r in [(0, 0.20, 0.9), (1, 0.30, 0.85), (2, 0.25, 0.8)]]
rows.append({"model": "reference", "nfe": 4, "fd": 0.15, "recall": 0.95})   # deterministic: no std
print(to_markdown(aggregate(rows), expected=3))
EOF
```

Output: `m` shows `0.25 ± 0.05` for fd and `0.85 ± 0.05` for recall; `reference` shows bare values. Most tables in the results document are single-seed; only the order study has seeds ("Open issues" there). Tutorial 4, Step 5 shows a case where the evaluation seed itself changed a number threefold.

## Failure modes we actually hit

- **Reward up, quality collapsed:** Steps 2 and 3, and the text run in tutorial 2 (vowel-only strings).
- **RL on a few-step student hacks harder** than RL on the teacher, because the 4-step policy is exact and high-variance. Results document, order study: reward 0.287 against 0.091, precision 0.32 against 0.80.
- **Order matters, neither order wins:** RL then distil keeps precision but loses recall (0.59), the results document says this fits distillation's mode-seeking compounding RL's collapse.
- **Unnormalised mixing of gradients** in `dmd2_reward`: check `reward_grad_ratio` first.

## Exercises

1. Why is `reward_std = 0` in a GRPO group a stop sign?
2. You see reward 0.6 and recall 0. What do you do?
3. Which reward modes work with `dmd2_reward` for a non-differentiable reward?

Answers:

1. Every advantage in that group is 0, so it contributes no gradient (tutorial 2). If it holds for all groups the policy has stopped learning, usually because it found a single output the reward likes.
2. Treat it as hacking. Add or raise `kl_coef`, lower the learning rate or epochs, and keep precision, recall and class accuracy on the dashboard next to the reward.
3. `mode: advantage`, which reweights the DMD2 gradient per sample by group-normalised reward. `mode: gradient` raises in `setup` for a reward without `differentiable = True`. Advantage mode has no run in the notes yet (D2).

Next: [6. Export and eval](06-export-and-eval.md).
