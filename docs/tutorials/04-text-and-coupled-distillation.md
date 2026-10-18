# 4. Text diffusion and coupled distillation

Goal: understand why a masked-diffusion language model gets bad when you cut its step count, what the `coupled` sampler and `coupled_kd` do about it, and how the per-position alternatives (KD, SDTT, `dcd`) behave. Run all of them on a tiny model.

Prerequisites: the setup block in [README.md](README.md) and [tutorial 1](01-denoising-as-a-policy.md) for the masked process. This tutorial trains its own text teacher.

## The idea in plain words

A masked diffusion LM starts from an all-mask sequence and, at each step, decodes some of the masked positions in parallel. Within one step, each decoded position is drawn independently from its own predicted distribution. With many small steps few tokens are decoded together, so the independence barely matters. With 1 to 4 steps, many tokens are decoded in the same step, and independent draws cannot reproduce dependencies between them.

The toy task makes this measurable. The corpus has a fixed set of lines, and the `valid` oracle counts a sample as valid only if it is exactly a corpus line (`packages/data/src/diffusion_omni/data/datasets.py`, `text_lines.oracle`). `coverage` is the share of corpus lines compatible with a prompt that were produced at least once.

The project's finding at toy scale ([docs/results/distillation_toy.md](../results/distillation_toy.md), summary item 1): a student that gets the teacher's **sampling noise** as an input (`coupled_kd`) fixes few-step decoding, and per-position methods barely move it. Per-position methods train the marginal distribution at each position. The failure is in the joint distribution, which no per-position target can change.

## The math, as implemented

**Coupled sampling** (`processes/masked.py`). The ancestral sampler uses fresh randomness at every step. The `coupled` sampler draws all of it once, up front, per trajectory:

- `unmask_time`: one uniform `tau` per position. A masked position is decoded in the step whose interval `[t_next, t)` contains `tau`. That happens with probability `(t - t_next)/t` given it was still masked at `t`, which is the ancestral unmask probability, so the process law is unchanged.
- `token_noise`: Gumbel noise per position and vocabulary entry, `(B, L, V)`. The token is `argmax(log p + g)`, an exact categorical sample (the Gumbel-max trick). With `coupled_noise: uniform` it is one uniform per position instead, and the token is chosen by inverting the CDF of `p` in token-id order: `(B, L)` noise instead of `(B, L, V)`.

So the whole K-step ancestral run is a **deterministic function of the noise**. The same noise gives the same text. `coupled_gumbel: 1` makes the output law exactly ancestral (test `test_coupled_masked_sampler_matches_ancestral_statistics`). Coupled steps return `log_prob=None`, so coupled rollouts cannot be RL-scored.

**`coupled_kd`** (`packages/distill/src/diffusion_omni/distill/methods/coupled_kd.py`).

1. Build a pool of teacher samples with the coupled sampler, keeping the noise (`TeacherPool._generate`).
2. The student is the teacher plus a noise input: `adapter.noise_proj`, a projection of `cond["token_noise"]` into the token embeddings, initialised to **zero**, so the student starts as the coupled teacher (`test_coupled_kd_student_starts_as_the_coupled_teacher`).
3. For a pool item with noise, pick a grid time `t` and rebuild the teacher's state there: positions with `unmask_time < t` are still masked, the others are already their final teacher tokens.
4. Loss: cross-entropy of the student's prediction on the masked positions against the teacher's final tokens.
5. The student samples with `coupled_gumbel: 0` (argmax): all diversity comes from the noise input, and the student can look at every position's noise through attention, so it can learn the teacher's joint sample for that noise. This is Luhman-style noise-to-sample distillation applied to masked diffusion.

`noise: gumbel` needs `L x V` noise per sample, which only fits small vocabularies (here V = 258 for the byte codec). `noise: uniform` feeds `L` scalars, expanded into `noise_features` (default 64) Fourier features.

**KD** (`methods/kd.py`) for text is cross-entropy on the teacher's final tokens at the masked positions of the states on the student grid. **SDTT** (`methods/sdtt.py`) trains in rounds where a student step equals two teacher steps, with targets given by the teacher's distribution at the sub-step where each token was decoded. **`dcd`** (`methods/dcd.py`, a single-mask discrete consistency ablation) is described in [docs/notes/D4.md](../notes/D4.md). All three are per-position.

## Walkthrough

### Step 1. A tiny text teacher

```bash
uv run diffusion-omni sft -c configs/text_masked_grpo.yaml "${TINY_TXT[@]}" \
  sft.steps=2000 sft.batch_size=32 sft.log_every=1000 output_dir="$OUT/txt"
```

About 20 seconds. The loss falls from about 1.0 at step 1000 to about 0.27 at step 2000 on the reference run.

### Step 2. Coupled sampling is a function of the noise

```bash
uv run python - "$OUT/txt/sft/final" <<'EOF'
import sys
from diffusion_omni.core import Pipeline
from diffusion_omni.core.sampling import sample
from diffusion_omni.core.utils import make_generator

pipe = Pipeline.from_checkpoint(sys.argv[1], device="cpu", overrides=["process.sampler=coupled"])
c = pipe.c
cond = c.prompts.encode(["the cat"] * 4, c.device)
shape = (4, *c.sample_shape)
noise = c.process.init_noise(shape, c.device, make_generator(0, c.device))
print({k: tuple(v.shape) for k, v in noise.items()})

def run(steps, noise):
    return sample(c.model, c.process, shape, {**cond, **noise}, steps)

a, b = run(16, noise), run(16, noise)
print("same noise, same output:", bool((a == b).all()))
other = c.process.init_noise(shape, c.device, make_generator(1, c.device))
print("different noise, same output:", bool((run(16, other) == a).all()))
for text in c.codec.decode(a):
    print(repr(text))
print("token agreement, 4 steps vs 16 steps, same noise:", float((run(4, noise) == a).float().mean()))
EOF
```

You will see noise shapes `unmask_time (4, 32)` and `token_noise (4, 32, 258)`; the same noise gives the same four strings; different noise gives different ones. The last line is the point of the method: the teacher's 4-step decode of the same noise agrees with its 16-step decode on most, but not all, tokens (0.75 on the reference run), and the student's job is to close the remaining gap.

### Step 3. Distil with each method

```bash
DT=(device=cpu teacher="$OUT/txt/sft/final" distill.pool_size=512 distill.teacher_steps=16
    distill.batch_size=32 distill.eval_every=100000 distill.log_every=200 distill.steps=600)

for m in kd coupled_kd sdtt dcd; do
  echo "== $m"
  uv run diffusion-omni distill -c configs/distill/text_masked_$m.yaml "${DT[@]}" output_dir="$OUT/t_$m" 2>&1 \
    | grep -E "^step=600|^saved" | cut -c1-260
done
```

It takes about 9 s for KD and `coupled_kd`, 16 s for SDTT and 22 s for `dcd`. Training logs per method: `kd_nll`; `ckd_nll` and `ckd_token_acc` (how often the student's argmax equals the teacher's token at masked positions: about 0.96 on the reference run); `sdtt_kl` and `round`; `dcd_kl`, `kl_to_teacher` and `mean_gap`. The `uniform` variant:

```bash
uv run diffusion-omni distill -c configs/distill/text_masked_coupled_kd.yaml "${DT[@]}" \
  distill.method.noise=uniform output_dir="$OUT/t_coupled_kd_uniform" 2>&1 | grep -E "^step=600|^saved" | cut -c1-200
```

### Step 4. Score them against a tuned teacher

```bash
uv run diffusion-omni scorecard --ref "$OUT/txt/sft/final" --ref-variant T0.6:process.temperature=0.6 \
  --ckpt kd="$OUT/t_kd/distill/final" --ckpt coupled_kd="$OUT/t_coupled_kd/distill/final" \
  --ckpt sdtt="$OUT/t_sdtt/distill/final" --nfe 1 2 4 8 -n 256 --ref-steps 16 --out "$OUT/sc_txt"
```

Columns for text: `diversity` (share of distinct strings within a prompt), `valid`, `coverage`, `reward`. The `reference[T0.6]` rows are the teacher sampled at temperature 0.6, the tuned baseline: lowering the temperature sharpens each position's draw and so also reduces the damage of independent draws. A student that does not beat that row has not shown more than a sampler setting.

**What the tiny run does and does not show.** The pipeline works, and `coupled_kd` runs with the zero-initialised noise input. It does not reproduce the separation reported in the results document. On our reference runs `coupled_kd` at 4 steps had `valid` of about 0.05 to 0.1 against 0.06 for the default teacher and 0.09 for `T0.6`, which is inside the noise of n = 256. The teacher here is small (its own `valid` at 16 steps was 0.17) and the students had 600 to 3000 steps. For the real result see the table in the results document: `valid` at 4 steps is 0.66 for `coupled_kd` against 0.20 for the tuned teacher (summary item 1), from the full-size configs (`dim 128, depth 4`, 3000 steps). We did not run that configuration for this tutorial.

### Step 5. Evaluate on noise the student has not trained on

`coupled_kd` students take the noise as input, so an evaluation that reuses training noise can overstate them. Check the effect on the 3000-step student before trusting a number:

```bash
DT3=(device=cpu teacher="$OUT/txt/sft/final" distill.pool_size=2048 distill.teacher_steps=16
     distill.batch_size=64 distill.eval_every=100000 distill.log_every=1000 distill.steps=3000)
uv run diffusion-omni distill -c configs/distill/text_masked_coupled_kd.yaml "${DT3[@]}" output_dir="$OUT/t_ckd_long" 2>&1 \
  | grep -E "^step=3000" | cut -c1-260

uv run python - "$OUT/t_ckd_long/distill/final" <<'EOF'
import sys, numpy as np
from diffusion_omni.core import Pipeline
from diffusion_omni.core.registry import DATASETS

p = Pipeline.from_checkpoint(sys.argv[1], device="cpu")
ds = DATASETS.build(p.c.cfg["sft"]["dataset"], **p.c.context)
prompts = p.c.prompts.sample(512, np.random.default_rng(0))
for seed in (0, 1, 3, 4, 5, 100):
    g = p.generate(prompts, seed=seed)
    print("generator seed", seed, {k: round(v, 3) for k, v in ds.oracle(g.samples, prompts).items()})
EOF
```

On the reference run, generator seed 0 gave `valid` 0.26 and every other seed gave 0.02 to 0.06. The in-training evaluation (`DistillTrainer.evaluate`) also samples with generator seed 0 on the config seed 0, and its `eval_valid` was 0.27 in that run; the scorecard samples the student with generator seed `--seed + 2`, which gave about 0.05 or lower. We did not establish the mechanism. The pool is built from a generator with the same seed, so overlap between evaluation noise and training noise is the first thing to check. Until that is settled, score `coupled_kd` students with the scorecard, with several `--seed` values, and not with `eval_valid`.

## Failure modes we actually hit

- **Per-position distillation does not fix the joint.** KD is roughly temperature sharpening and SDTT degrades every round in the results document (summary item 1). `dcd` is indistinguishable from the undistilled teacher in [docs/notes/D4.md](../notes/D4.md): a single-seed negative result, with the argument that a factorised single-mask target has the teacher as its fixed point.
- **Not comparing to a tuned teacher.** The `T0.6` row is the right baseline (also in `scripts/distill_toy_report.py`).
- **`coupled_kd` needs `distill.data: teacher`.** It distils the teacher's own noise-to-sample map and raises otherwise.
- **Gumbel noise is `L x V`.** Use `noise: uniform` for a real vocabulary. The results document reports that on the toy it nearly matches the Gumbel version, with validity 0.35 / 0.49 / 0.64 / 0.77 at 1 / 2 / 4 / 8 steps, and lists a real tokenizer and model as open.
- **Coupled rollouts are not RL policies** (`log_prob=None`).
- **Text students are not RL-able through `coupled`.** The `renoise` students of tutorial 3 are Gaussian policies; a coupled student is a deterministic function of the noise.

## Exercises

1. For `L = 32`, `V = 258` and a hypothetical `V = 50000`, how many noise floats per sample does `noise: gumbel` need, and how many does `noise: uniform` need?
2. Why is the unmask probability `(t - s) / t` rather than `t - s`?
3. In `CoupledKD.loss`, which positions are masked in the rebuilt state at time `t`?
4. A coupled student at step count 1 has `t_next = 0`. Which positions does it decode in that single step?

Answers:

1. Gumbel: `32 * 258 = 8256` and `32 * 50000 = 1,600,000`. Uniform: `32` in both cases (plus the 32 unmask times, and the `noise_features` expansion happens inside the model).
2. The forward process masks a position with probability `t`, so a position that is still masked at `t` was unmasked-by-time `s` with probability `(t - s)/t` given that. The reverse step must reproduce that conditional.
3. Those with `unmask_time < t` (and not prefix positions). A position is decoded in the interval `[t_next, t)` containing its `unmask_time`, so at time `t` the positions with larger `unmask_time` are already decoded and the ones with smaller are not.
4. All free (non-prefix) positions: the interval is `[0, 1)`, which contains every `unmask_time`.

Next: [5. Rewards, reliability and hacking](05-rewards-reliability-hacking.md).
