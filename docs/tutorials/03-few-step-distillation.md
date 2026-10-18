# 3. Few-step distillation

Goal: know what each distillation method in `diffusion-omni-distill` optimises, train each one on a tiny teacher, score them against the right baseline, and reproduce on a small scale the one failure that cost us the most time (the flow-map time-embedding jitter).

Prerequisites: the setup block in [README.md](README.md) and the image teacher from [tutorial 1](01-denoising-as-a-policy.md), Step 1 (`$OUT/img/sft/final`).

## The idea in plain words

A teacher needs 10 to 50 network calls to make a sample. A distilled student makes a similar sample in 1 to 4. Methods differ in what the student is trained to match:

- the teacher's trajectory endpoints (KD, progressive),
- the teacher's own consistency along a trajectory (consistency, flow map),
- the teacher's output distribution, without pairing individual samples (DMD2).

The question that decides whether a method is worth using is not "does it work", it is "does it beat the teacher with a tuned sampler at the same number of steps". Few-step failures of a teacher are often sampler settings, so the scorecard (tutorial 6) always puts tuned-teacher rows next to the students.

## Setting

`diffusion-omni distill -c configs/distill/<name>.yaml` loads a frozen teacher from `teacher:` (a checkpoint directory), builds a student from the teacher's model spec and weights, and trains it with a registered `Distiller` (`packages/distill/src/diffusion_omni/distill/trainer.py`, `methods/`). Teacher samples and their intermediate states on the student's grid are generated once into a pool (`distill.data: teacher`, `distill.pool_size`, `distill.teacher_steps`), or real data is used. The student is exported with its sampler and its timestep grid in `config.yaml`, so `diffusion-omni sample` and the scorecard use it unchanged.

Notation: `t` is time (1 is noise), `x_t = alpha(t) x0 + sigma(t) noise`, `to_x0(pred, x_t, t)` converts a network output to an `x0` estimate. For flow matching `alpha = 1 - t`, `sigma = t`, and `pred` is a velocity `v`.

## The methods, as implemented

**KD** (`methods/kd.py`). Take teacher ODE states `x_t` on the student's grid (4 steps by default) and regress `to_x0(student(x_t, t))` onto the trajectory's endpoint `x0` with MSE. Simple, and the baseline the others must beat. A regression onto a conditional mean blurs as the step count falls.

**Consistency** (`methods/consistency.py`). Let `f(x, t) = to_x0(student(x, t))`. For adjacent times `t_hi > t_lo` on a 32-interval grid: noise a data sample to `x_hi`, let the teacher take one ODE step to `x_lo`, and train

```text
f(x_hi, t_hi)  ->  stopgrad f(x_lo, t_lo)
loss = pseudo-Huber(distance), weighted by 1 / (t_hi - t_lo)
```

The target is the current student under stop-gradient (the iCT choice) or an EMA copy (`target: ema`). The boundary condition `f(x, 0) = x` holds for both continuous processes. Sampler: `renoise`.

**DMD2** (`methods/dmd2.py`). Distribution matching, no pairing. Three networks: the frozen teacher (the "real" score), a student, and a "fake score" network that starts as a copy of the teacher and is trained online to denoise the student's samples. For a student sample `x0_g`, noise it to `x_tau`, and compute

```text
grad = ( x0_fake(x_tau) - x0_real(x_tau) ) / mean| x0_g - x0_real(x_tau) |
loss = 0.5 * || x0_g - stopgrad(x0_g - grad) ||^2
```

so the student's gradient is the difference between the fake and real score (a test, `test_dmd2_generator_gradient_is_the_score_difference`, pins this). The schedule is `["fake"] * fake_updates + ["student"]`: five fake-score updates per student update in the shipped config. "Backward simulation" means the student is trained on its own intermediate states: the generator starts from the state the student itself reaches at a random grid point. Sampler: `renoise`.

**Flow map** (`methods/flow_map.py`). The student is a two-time network `u(x_t, t, s)`, the average velocity over `[s, t]`, so one jump is `x_s = x_t + (s - t) u`. Differentiating `(t - s) u = integral of v` along the teacher's ODE gives the MeanFlow training target

```text
u_target = v(x_t, t) - (t - s) * d/dt u(x_t, t, s)
d/dt u   = v . grad_x u + partial_t u        (one forward-mode JVP, under stop-gradient)
```

with `v` from the teacher. The extra `s` input comes from an `s_embed` layer with zero-initialised output, so the student **starts as the teacher's Euler sampler** (test: `test_flow_map_student_starts_as_teacher_euler`). One student serves any step count. JVPs need the non-fused attention path, which the method switches on (`math_attention`).

**Progressive** (`methods/progressive.py`, see [docs/notes/D3.md](../notes/D3.md)). Rounds that halve the step count: in each round the student learns to match two ODE steps of the previous round's teacher in one step, using a target computed from `alpha_sigma` only. At each round boundary the student's EMA becomes the new frozen teacher. `rounds: 3` from `start_steps: 16` gives 8, 4, then 2 steps.

**rCM** (`methods/rcm.py`) is a flow map with `s_mode: zero` (a continuous-time consistency model) plus a DMD2 score regulariser. `dmd2_reward` (tutorial 5) and the masked-text methods (tutorial 4) are covered elsewhere.

The full method table is in [packages/distill/README.md](../../packages/distill/README.md).

## Walkthrough

### Step 1. Train every method for 300 steps

The shipped configs run 600 to 2000 steps on a dim-128 model. These overrides shrink the pool, teacher steps, batch and step count. The teacher is the tiny one from tutorial 1.

```bash
DIST=(device=cpu teacher="$OUT/img/sft/final" distill.pool_size=256 distill.teacher_steps=8
      distill.batch_size=16 distill.eval_every=100000 distill.log_every=100 distill.steps=300)

for m in kd consistency dmd2 flow_map rcm; do
  echo "== $m"
  uv run diffusion-omni distill -c configs/distill/image_flow_$m.yaml "${DIST[@]}" output_dir="$OUT/d_$m" 2>&1 \
    | grep -E "^step=300|^saved"
done
uv run diffusion-omni distill -c configs/distill/image_flow_progressive.yaml "${DIST[@]}" \
  distill.method.rounds=2 output_dir="$OUT/d_progressive" 2>&1 | grep -E "^step=300|^saved"
```

Wall-clock on the reference run: KD 3 s, consistency 4 s, flow map 5 s, progressive 5 s, DMD2 17 s, rCM 20 s. `distill.method.rounds=2` is needed for progressive because 300 steps cannot cover the config's 3 rounds (`setup` raises if `steps < rounds`; the test suite uses `rounds=2` for the same reason).

Each method logs its own loss. What to look for at step 300:

| method | logged | reading it |
|---|---|---|
| `kd` | `kd_mse` | falls steadily (about 0.003 at step 300 on the reference run) |
| `consistency` | `cd_distance` | falls (to about 0.2 by step 300); gradient norm is large (tens), which is how this loss behaves here |
| `dmd2` | `fake_denoise`, `dm_loss`, `dm_grad` | `dm_loss` is not a quality measure: the target moves as the fake score trains. Judge DMD2 by samples |
| `flow_map` | `fm_mse`, `fm_gap` | `fm_mse` is a self-referential residual (the target uses the student's own derivative), see the failure mode below |
| `progressive` | `progressive_loss`, `round`, `student_steps` | `student_steps` counts down as rounds advance (8, 6, 4 in the reference run with `rounds=2`) |

`eval_*` fields on the last line are the in-training check (`DistillTrainer.evaluate`), on the student's trained grid.

### Step 2. Score the students against tuned-teacher baselines

```bash
uv run diffusion-omni scorecard --ref "$OUT/img/sft/final" \
  --ref-variant shift3:process.shift=3 --ref-variant renoise:process.sampler=renoise \
  --ckpt kd="$OUT/d_kd/distill/final" --ckpt consistency="$OUT/d_consistency/distill/final" \
  --ckpt dmd2="$OUT/d_dmd2/distill/final" --ckpt flow_map="$OUT/d_flow_map/distill/final" \
  --nfe 1 2 4 -n 128 --ref-steps 16 --out "$OUT/sc_img"
```

It prints a Markdown table and writes `$OUT/sc_img.md` and `.json`. Rows:

- `reference@16`: the teacher with its ODE sampler at 16 steps. Teacher-relative `fd`, `precision` and `recall` are measured against this row.
- `reference`, `reference[shift3]`, `reference[renoise]`: the same teacher weights at 1, 2 and 4 steps with the default grid, a flow time shift of 3, and the few-step `renoise` sampler. These are the baselines a student has to beat.
- one row per student and NFE.

Read three columns together: `fd` (distance to the reference samples, lower is better), `precision` (are samples inside the reference manifold) and `recall` (does the student cover the reference). Metric definitions are in `packages/reward/src/diffusion_omni/reward/metrics.py`. They use a random feature projection, so they rank samplers on the same data and are not comparable with Inception FID.

What to expect with these tiny models: the numbers are noisy (n = 128, 300 training steps, one seed) and the students are not better than the teacher. What the table does show is mechanical:

- `flow_map` rows are nearly identical to the `reference` (Euler) rows at the same NFE. That is the zero-initialised `s_embed` at work: after 300 steps the student has barely left the teacher's Euler sampler.
- `dmd2` and `consistency` rows have higher precision and lower recall than the `reference` rows as NFE grows (mode-seeking), for example recall falls with NFE for `dmd2` on the reference run.
- A student uses its trained grid only when NFE equals the grid length (4 for `kd`, `consistency`, `dmd2` and `flow_map` here). At other NFE the scorecard uses the process's default grid, whose times the student did not train on.

At real toy scale the picture is in [docs/results/distillation_toy.md](../results/distillation_toy.md): at 4 or more steps the teacher with a tuned sampler beats every student, and students win only at 1 to 2 steps (summary item 2). That document also shows the quality-versus-coverage trade (summary item 3).

### Step 3. Reproduce the flow-map failure

The shipped configs train the teacher with the default `1000 t` time embedding. Check what that does, and what `time_scale: 10` changes. First a second tiny teacher using the shipped `_ts10` config:

```bash
uv run diffusion-omni sft -c configs/image_flow_grpo_ts10.yaml "${TINY_IMG[@]}" \
  sft.steps=600 sft.batch_size=32 sft.log_every=300 output_dir="$OUT/img_ts10"
```

Now measure how smooth each teacher's velocity is in `t`. The script reports a finite-difference acceleration along the teacher's own ODE for shrinking step sizes `h`:

```bash
uv run python scripts/flow_map_diagnose.py --smoothness "$OUT/img/sft/final" "$OUT/img_ts10/sft/final" -n 64 2>&1 | grep "h="
```

On the reference run:

```text
.../img/sft/final       h=0.1: 1.06 h=0.03: 1.8 h=0.01: 2.86 h=0.003: 8.47 h=0.001: 11.6
.../img_ts10/sft/final  h=0.1: 0.765 h=0.03: 0.671 h=0.01: 0.654 h=0.003: 0.648 h=0.001: 0.647
```

For the default teacher the acceleration grows without bound as `h` shrinks, which means the velocity field jitters in `t`. For the `time_scale: 10` teacher it converges. The same pattern, at the repo's toy scale, is the table in [docs/results/distillation_toy.md](../results/distillation_toy.md#flow-maps-need-a-teacher-that-is-smooth-in-t).

Why it matters: a JVP measures the time derivative exactly, jitter included, so the MeanFlow target is mostly noise. Train a flow map from each teacher with the same config:

```bash
for t in img img_ts10; do
  echo "== teacher $t"
  uv run diffusion-omni distill -c configs/distill/image_flow_flow_map.yaml "${DIST[@]}" \
    teacher="$OUT/$t/sft/final" output_dir="$OUT/fm_$t" 2>&1 | grep -E "^step=(100|300) |time_scale" | cut -c1-120
done
```

Expected: a `UserWarning` for the default teacher (`flow_map teacher has time_scale=1000 ...`), and `fm_mse` of about 1.1 for it, against about 0.05 for the `ts10` teacher. The tiny models reproduce the effect. The large gap in `fm_mse` is the symptom; the sample-quality consequences at scale are in the results document.

## Failure modes we actually hit

- **Time-embedding jitter breaks flow-map JVPs.** The default `model.time_scale: 1000` makes the teacher's velocity jitter in `t`; the residual never falls. Fix: teach the teacher with `model.time_scale: 10` (config `_ts10`). Every image method in the results document distils better from the smooth teacher, and the shipped configs keep 1000 so that published tables reproduce ([docs/results/distillation_toy.md](../results/distillation_toy.md#flow-maps-need-a-teacher-that-is-smooth-in-t)). `flow_map` warns above `time_scale` 100; `rcm` does not.
- **Comparing against an untuned teacher.** The 1-step and 2-step teacher rows improve a lot with a time shift or the `renoise` sampler. Use the best `reference[...]` row, not the default one.
- **DMD2 stage sampling.** DMD2 trains the student at a random grid stage, but inference returns the last stage. With uniform stage draws the last stage can drift; `last_stage_prob` (0.5 in the shipped config) draws it more often. Rationale and a short CPU check are in [docs/notes/D1.md](../notes/D1.md); that check is single-seed. Try `distill.method.last_stage_prob=null` to get the old behaviour.
- **DMD2 is a discrete-time student.** It is best near its trained grid; off-grid step counts are untrained times. In the results document DMD2 is best at 2 steps, not at the 4-step grid it was trained on, and it degrades at 8 steps ([summary item 5](../results/distillation_toy.md#summary)).
- **DMD2 on the Gaussian (VP, epsilon) teacher fails** in the results document, where the DDPM teacher is itself weak.
- **Students are fixed-grid.** KD and DMD2 degrade away from their grid; the flow map improves with NFE (results document, "One flow-map student covers every step count").
- **Progressive on a Gaussian teacher with the `max(SNR, 1)` weight** produced clipped gradients in the reviewer's runs; the config uses `weighting: snr` ([docs/notes/D3.md](../notes/D3.md)).

## Exercises

1. A flow process with `shift = 3` has a 4-step grid. What is it?
2. After the flow-map training in Step 1 (300 steps), why are its scorecard rows close to the `reference` rows?
3. In consistency distillation, why is `f(x, 0) = x` automatic for flow matching?
4. Why does the DMD2 loss `0.5 * || x0_g - stopgrad(x0_g - grad) ||^2` produce gradient `grad` on `x0_g`?

Run this to check 1 and 3:

```bash
uv run python - <<'EOF'
import torch
from diffusion_omni.core.registry import PROCESSES
import diffusion_omni.core.processes  # registers the built-in processes

p = PROCESSES.build({"type": "flow", "shift": 3.0})
print("shift 3 grid:", [round(float(t), 3) for t in p.timesteps(4, "cpu")])
x_t, pred = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
print("to_x0 at t=0 equals x_t:", bool(torch.equal(p.to_x0(pred, x_t, torch.zeros(2)), x_t)))
EOF
```

Answers:

1. `shift * t / (1 + (shift - 1) * t)` applied to `linspace(1, 0, 5)` gives `[1, 0.9, 0.75, 0.5, 0]`. The grid spends more of its steps at high noise.
2. The student starts as the teacher's Euler sampler because of the zero-initialised `s_embed`, and 300 steps leave it close to that start. The results document reports the same thing at full length for the default teacher: the student matched the teacher's Euler sampler at every step count.
3. `to_x0 = x_t - t * pred` for flow, so at `t = 0` it returns `x_t` whatever the network outputs.
4. The gradient of `0.5 * mse_loss(a, stopgrad(a - g))` with respect to `a` is `g / numel` (`F.mse_loss` averages). The code builds `a - g` as the target on purpose, so the backward pass delivers the score difference to the student, and the network that computed `a` receives it through the usual chain rule.

Next: [4. Text diffusion and coupled distillation](04-text-and-coupled-distillation.md).
