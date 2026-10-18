# 1. Denoising as a policy

Goal: see why a diffusion sampler can be treated as a stochastic policy with exact log-probabilities, and check that in code. Everything later (RL in tutorial 2, RL on few-step students in tutorial 3) depends on this.

Prerequisite: the setup block in [README.md](README.md).

## The idea in plain words

A sampler starts from noise (or from a fully masked sequence) at time `t = 1` and walks down a list of times to `t = 0`, which is data. At each step the network output is turned into a distribution over the next state, and the sampler draws from it. That draw is an action. The state is `(x_t, t, condition)`. A whole sample is one episode, and a reward arrives only at the end.

If the step distribution has a known density, then for any stored `(x_t, x_next)` pair you can compute `log pi(x_next | x_t)` under the current network. That is what policy-gradient RL needs. In this repo one function does both jobs: `process.transition(pred, x_t, t, t_next, x_next=None)` in `packages/core/src/diffusion_omni/core/processes/base.py`.

- Without `x_next` it draws an action (sampling).
- With `x_next` it scores that action (RL re-scoring).

It returns a `Transition` with `x_next`, a per-sample `log_prob` (or `None` for a deterministic step), and the distribution parameters in `stats`, which `process.kl(new, ref)` uses.

Not every sampler is a policy. An ODE sampler is deterministic, so its `log_prob` is `None`, and RL refuses to run on it.

## The math, as implemented

All three processes use `t = 1` for noise and `t = 0` for data.

**Flow matching** (`processes/flow.py`). Noising is `x_t = (1 - t) x0 + t * noise`, the network predicts the velocity `v ~ noise - x0`, and the SFT loss is the MSE between prediction and `noise - x0`. The default `sde` sampler is the reverse SDE that has the same marginals as the ODE. With `a = noise_level`:

```text
sigma = a * sqrt(t / (1 - t))
mean  = x_t + dt * ( v + sigma^2 / (2t) * (x_t + (1 - t) v) )      dt = t_next - t  (negative)
std   = sigma * sqrt(-dt)
x_next ~ N(mean, std^2 I)
```

So each step is a diagonal Gaussian, and `log_prob` is the Normal log-density of `x_next`, reduced over dimensions. At `t = 1` the factor `1/(1-t)` blows up, so the code evaluates it at `t_next` there. With `noise_level <= 0` the `sde` sampler falls back to `ode` (Euler: `x + dt * v`) and the step has no log-prob.

**Gaussian (VP) diffusion** (`processes/gaussian.py`). The network predicts epsilon. The `ddim` sampler with `eta > 0` is a Gaussian step with mean built from the predicted `x0` and `eps`, and variance `eta^2 * (1 - ab_s) / (1 - ab_t) * (1 - ab_t / ab_s)`. `eta = 0` is deterministic.

**The renoise sampler** (`ContinuousProcess.renoise_transition`). Predict `x0`, then draw `x_next ~ N(alpha(t_next) x0_hat, sigma(t_next)^2 I)`. This is the multistep sampler of consistency models and DMD2, and it is also a Gaussian policy. The last step has `sigma(0) = 0`, so it is deterministic and has no log-prob.

**Masked (absorbing) diffusion** (`processes/masked.py`). A position is masked with probability `t`. A reverse step from `t` to `s` unmasks each masked position with probability `p = (t - s) / t` and draws its token from the model's distribution. Decoded tokens never change. The log-prob is exact:

```text
log pi = sum over positions unmasked now  [ log p + log p_token ]
       + sum over positions still masked  [ log(1 - p) ]
```

Prompt positions (`cond["prefix"]`) are fixed from the start.

**The reduction.** `FlowMatching` and `GaussianDiffusion` default to `reduction: mean`, so their `log_prob` is the mean over dimensions, not the sum. `MaskedDiffusion` defaults to `sum`. The RL ratio is `exp(logp - old_logp)`, so with `mean` the log-ratio is the sum's log-ratio divided by the number of dimensions. The image configs set `clip_range: 1e-4`, which is a very tight clip on that tempered ratio.

## Walkthrough

### Step 1. Train a tiny image teacher

```bash
uv run diffusion-omni sft -c configs/image_flow_grpo.yaml "${TINY_IMG[@]}" \
  sft.steps=600 sft.batch_size=32 sft.log_every=200 output_dir="$OUT/img"
```

Expected: three log lines whose `loss` falls (about 0.42 at step 400 and 0.27 at step 600 on the reference run), then `saved .../img/sft/final`. It takes a few seconds.

### Step 2. Sample, then re-score a stored action

```bash
uv run python - "$OUT/img/sft/final" <<'EOF'
import sys, torch
from diffusion_omni.core import Pipeline
from diffusion_omni.core.sampling import step

pipe = Pipeline.from_checkpoint(sys.argv[1], device="cpu")
c = pipe.c
print("process:", type(c.process).__name__, "sampler:", c.process.sampler)

# A trajectory of the SDE policy: 4 prompts, 4 steps.
gen = pipe.generate(prompts=[1, 1, 2, 2], seed=0, return_trajectory=True)
traj = gen.trajectory
print("xs", tuple(traj.xs.shape), "log_probs", tuple(traj.log_probs.shape))
print("timesteps", [round(float(t), 3) for t in traj.timesteps])
print("stochastic", traj.stochastic.tolist())

# Re-score step k with the stored x_next: this is what the RL trainer does.
cond = c.prompts.encode(gen.prompts, c.device)
k = 1
with torch.no_grad():
    tr = step(c.model, c.process, traj.xs[:, k], traj.timesteps[k], traj.timesteps[k + 1], cond,
              c.sampling.guidance_scale, x_next=traj.xs[:, k + 1])
print("stored   logp:", [round(v, 4) for v in traj.log_probs[:, k].tolist()])
print("rescored logp:", [round(v, 4) for v in tr.log_prob.tolist()])
print("max |diff|:", float((tr.log_prob - traj.log_probs[:, k]).abs().max()))

# The same number from torch.distributions.
mean, std = tr.stats["mean"], tr.stats["std"]
ref = torch.distributions.Normal(mean, std).log_prob(traj.xs[:, k + 1]).mean(dim=(1, 2, 3))
print("torch Normal :", [round(v, 4) for v in ref.tolist()])

# KL of a policy with itself is zero.
print("KL(new,new):", c.process.kl(tr, tr).tolist())

# An ODE step has no log-prob.
c.process.sampler = "ode"
with torch.no_grad():
    tr_ode = step(c.model, c.process, traj.xs[:, k], traj.timesteps[k], traj.timesteps[k + 1], cond, 1.0)
print("ode log_prob:", tr_ode.log_prob)
EOF
```

What to look at in the output:

- `xs` is `(4, 5, 3, 16, 16)`: the prior plus one state per step. `log_probs` is `(4, 4)`: one value per step.
- The stored and re-scored log-probs are identical (`max |diff|: 0.0`), and they equal the `torch.distributions.Normal` value averaged over dimensions. That equality is what the test `test_continuous_log_prob_matches_torch_normal` pins.
- `KL(new,new)` is exactly zero.
- The ODE step returns `None`.

### Step 3. The same check for masked text

```bash
uv run diffusion-omni sft -c configs/text_masked_grpo.yaml "${TINY_TXT[@]}" \
  sft.steps=300 sft.batch_size=16 sft.log_every=100 output_dir="$OUT/txt_small"

uv run python - "$OUT/txt_small/sft/final" <<'EOF'
import sys, torch
from diffusion_omni.core import Pipeline
from diffusion_omni.core.sampling import step

pipe = Pipeline.from_checkpoint(sys.argv[1], device="cpu")
c = pipe.c
gen = pipe.generate(prompts=["the cat", "a dog"], seed=0, return_trajectory=True, num_steps=4)
traj = gen.trajectory
for k in range(traj.xs.shape[1]):
    t = float(traj.timesteps[k])
    print(f"t={t:.2f} masked positions per sample:", (traj.xs[:, k] == c.process.mask_id).sum(1).tolist())
cond = c.prompts.encode(gen.prompts, c.device)
with torch.no_grad():
    tr = step(c.model, c.process, traj.xs[:, 0], traj.timesteps[0], traj.timesteps[1], cond, 1.0,
              x_next=traj.xs[:, 1])
print("stored  :", [round(v, 3) for v in traj.log_probs[:, 0].tolist()])
print("rescored:", [round(v, 3) for v in tr.log_prob.tolist()])
print("KL(new,new):", c.process.kl(tr, tr).tolist())
print(gen.samples)
EOF
```

The masked-position count falls roughly as `t` (the prompt characters are never masked, so it starts below 32), reaches 0 at `t = 0`, and the re-scored log-probs again match the stored ones. The sampled text is poor because the model is tiny; that is expected.

## Failure modes we actually hit

- **A deterministic sampler cannot be RL-tuned.** `RLTrainer.rollout` raises `sampler is deterministic (eta=0 / noise_level=0); RL needs a stochastic policy` when no step has a log-prob. Config checks to make: `process.noise_level > 0` for flow, `eta > 0` for Gaussian.
- **The last renoise step has no log-prob.** For a few-step student with the `renoise` sampler, `traj.stochastic` is `[True, True, True, False]` on a 4-step grid (verified on a tiny DMD2 student). Only the first three steps are trained by RL.
- **Coupled masked sampling cannot be RL-scored.** The `coupled` sampler (tutorial 4) returns `log_prob=None` by design (`docs/ARCHITECTURE.md`).
- **`mean` versus `sum`.** Do not compare log-probs across processes without checking `reduction`.

## Exercises

1. Why is `traj.log_probs` zero at a deterministic step, while `transition` returns `None`?
2. Re-run the Step 2 script with `process.reduction=sum` (pass `overrides=["process.reduction=sum"]` to `Pipeline.from_checkpoint`). By what factor do the log-probs change?
3. For a masked step from `t` to `t_next`, what is the probability that a masked position stays masked?
4. A flow step at `t = 1` has `1/(1-t) = infinity`. Which line of `FlowMatching.transition` avoids it?

Answers:

1. `sample()` stacks `log_probs` into one tensor, so it writes `torch.zeros` for steps whose `log_prob` is `None`, and records which steps are real in `traj.stochastic` (`core/sampling.py`). Always mask with `traj.stochastic`.
2. By exactly the number of dimensions per sample: the same Normal log-densities are summed instead of averaged, and `3 * 16 * 16 = 768`.
3. `1 - p = t_next / t`. In the code that is the `stayed * (1 - p).log()` term.
4. `sigma = self.noise_level * (t / (1 - torch.where(t >= 1, t_next, t))).sqrt()` evaluates the denominator at `t_next` when `t >= 1`.

Next: [2. From SFT to RL](02-sft-to-rl.md).
