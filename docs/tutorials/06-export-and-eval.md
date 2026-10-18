# 6. Export and eval

Goal: read a scorecard correctly, export a student as a framework-neutral bundle, load it back, and know what the multi-seed report script does.

Prerequisites: the setup block in [README.md](README.md), the students `$OUT/d_dmd2` from [tutorial 3](03-few-step-distillation.md) and `$OUT/t_coupled_kd` from [tutorial 4](04-text-and-coupled-distillation.md).

Coming, not covered: the `eval` command, the batched reward service and the miles adapter.

## Scorecard anatomy

`diffusion-omni scorecard` (`packages/reward/src/diffusion_omni/reward/scorecard.py`) works in this order:

1. Fix the prompts (`-n` of them, seeded by `--seed`).
2. Generate the reference's full-step samples (`--ref-steps`, ODE sampler for continuous processes). Teacher-relative metrics use these.
3. If the reference config names an SFT dataset, also draw real data: `data_*` metrics, and the dataset's exact oracle (`class_acc`, `shape_err`, `valid`, `coverage`, ...).
4. Score the reference itself at each NFE, then each `--ref-variant name:key=value[,key=value]` (the same weights with config overrides), then each `--ckpt name=dir`.
5. A checkpoint uses its own trained grid only if NFE equals the grid length; otherwise the process default grid with that many steps.

Metrics: `fd` (Frechet distance in a seeded random feature space), `precision` and `recall` (k-NN manifold, k = 3), `diversity` (within-prompt distance for arrays, share of distinct strings for text), `reward`, `ms_per_sample`. The feature space is random ReLU features, which ranks samplers against each other and is not comparable to Inception FID (`metrics.py`).

## Walkthrough

### Step 1. Export a bundle and load it

```bash
uv run diffusion-omni export --ckpt "$OUT/d_dmd2/distill/final" --out "$OUT/bundle_img"
ls "$OUT/bundle_img"
```

You get `model.safetensors` (or `model.pt` if safetensors is not installed), `model.json`, `sampler.json`, `README.md`, and `resources/` when the codec or prompts reference files. LoRA is merged. `sampler.json` carries the sampler (`renoise` for this DMD2 student), the explicit `timesteps` grid, `guidance_scale`, `sample_shape`, the codec and prompt specs, the weights' sha256 and `distilled_from` provenance. The schema is in [docs/notes/D5.md](../notes/D5.md) and in each bundle's README.

```bash
uv run python - "$OUT/d_dmd2/distill/final" "$OUT/bundle_img" <<'EOF'
import sys, numpy as np
from diffusion_omni.core import Pipeline
from diffusion_omni.distill.export import load_bundle

ckpt, bundle = sys.argv[1:3]
a = load_bundle(bundle, device="cpu").generate(prompts=[1, 2], seed=0).samples
b = Pipeline.from_checkpoint(ckpt, device="cpu").generate(prompts=[1, 2], seed=0).samples
print(a.shape, "identical to the checkpoint:", np.array_equal(a, b))
EOF
```

Expected: `(2, 3, 16, 16) identical to the checkpoint: True`. `load_bundle` checks the weight hash and rejects a malformed `sampler.json` with an error naming the key. It imports any plugin modules named in `sampler.json`, so load only bundles you trust.

The same works for a text student with the coupled sampler:

```bash
uv run diffusion-omni export --ckpt "$OUT/t_coupled_kd/distill/final" --out "$OUT/bundle_txt"
uv run python - "$OUT/bundle_txt" "$OUT/t_coupled_kd/distill/final" <<'EOF'
import sys
from diffusion_omni.core import Pipeline
from diffusion_omni.distill.export import load_bundle

a = load_bundle(sys.argv[1], device="cpu").generate(prompts=["the cat"], seed=0).samples
b = Pipeline.from_checkpoint(sys.argv[2], device="cpu").generate(prompts=["the cat"], seed=0).samples
print(a, a == b)
EOF
```

### Step 2. Check the exported grid

```bash
python3 -c "
import json, sys
m = json.load(open(sys.argv[1] + '/sampler.json'))
print(m['sampler'], m['timesteps'], m['num_steps'], m['distilled_from']['teacher'])
" "$OUT/bundle_img"
```

Output: `renoise [1.0, 0.75, 0.5, 0.25, 0.0] 4 sft/final`. The teacher is stored as its last two path components only.

### Step 3. The multi-seed report, dry run

`scripts/distill_toy_report.py` trains every student per seed against fixed teachers, scores them with identical prompts, and aggregates (details in [docs/notes/D6.md](../notes/D6.md)). It expects the shipped teachers under `runs/`, which this tutorial does not create, so only the plan can be shown here:

```bash
uv run python scripts/distill_toy_report.py --seeds 0 1 --groups text_masked_grpo --dry-run | tail -n 3
```

It prints the runs it would train (`train text_masked_dcd seed=1 -> runs/distill/text_masked_dcd_s1`) and `8 to train, 0 stale, 0 done, 0 reused legacy`, and creates nothing. We did not run the real report; its output is the document [docs/results/distillation_toy.md](../results/distillation_toy.md). Rules to know: a run is skipped only if its `provenance.json` matches, a stale run stops the script unless `--retrain-stale`, and cells fed by fewer than the planned seeds are marked `(k/N)`.

## Reading a scorecard row, a checklist

1. Compare the student with the best `reference[...]` row at the same NFE, not the default one.
2. Read `fd` with `precision` and `recall`: low `fd` with low recall is a narrow student.
3. Check the oracle columns (`class_acc`, `valid`, ...) against the reward.
4. Is the NFE on the student's trained grid?
5. How many seeds, how many samples?

## Failure modes we actually hit

- **Scoring off-grid.** A 4-step DMD2 student at 8 steps visits untrained times (results document, summary item 5).
- **Stale students.** A checkpoint trained before a code change is silently reused by hand-written scripts; the report script's provenance check exists for this reason (D6).
- **Seed-0 evaluation of noise-conditioned students:** see tutorial 4, Step 5.

## Exercises

1. What happens to `sampler.json` if you export an RL checkpoint rather than a distilled one?
2. Why does the bundle store a sha256?
3. Run `diffusion-omni scorecard` on `$OUT/bundle_img`. Does it work?

Answers:

1. There is no `distilled_from`, since that block is optional; the sampler and grid come from the checkpoint's config. We did not run this case.
2. `load_bundle` reads exactly the named weights file and verifies the hash, so a swapped or corrupted file fails loudly.
3. No: the scorecard takes checkpoint directories (`config.yaml` plus `model.pt`), not bundles. Use the original checkpoint.
