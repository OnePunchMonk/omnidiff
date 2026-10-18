# Guide: eval-scorecard

Compare students with their teacher across step counts, on one fixed prompt set, in one table. Each row is scored against the teacher (teacher-relative FD, precision, recall, diversity), against real data when the reference config names a dataset, and by the dataset's exact oracle and the reward when they exist.

Needs a teacher and at least one student (see [sft](sft.md) and [distill](distill.md)).

## Tiny run (CPU)

```bash
uv run diffusion-omni scorecard --ref runs/guide/img/sft/final --ckpt dmd2=runs/guide/dmd2/distill/final --ref-variant ode:process.sampler=ode --nfe 1 2 4 --n 16 --ref-steps 8 --device cpu --out runs/guide/scorecard
```

Writes `runs/guide/scorecard.md` (the table) and `runs/guide/scorecard.json` (the rows, for further analysis). The table is also printed.

## Full run

```bash
uv run diffusion-omni scorecard --ref runs/image_flow_grpo/sft/final --ckpt dmd2=runs/distill/image_flow_dmd2/distill/final --nfe 1 2 4 8 --out runs/scorecard
```

## Options

- `--ckpt name=dir` (repeatable): students to score.
- `--ref-variant name:key=value[,key=value]` (repeatable): a tuned teacher baseline. The same weights with a different sampler or time shift. Compare students with the best tuned teacher row, not only the default one. Example: `--ref-variant T0.7:process.temperature=0.7` for masked text.
- `--nfe`: step counts. When a count matches the student's trained grid, that grid is used; otherwise the process default grid is used.
- `-n`: samples per row (default 256). `--ref-steps`: teacher steps for the reference row (default 32). `--seed`.

## Reading the table

| column | meaning |
|---|---|
| `diversity` | within-prompt diversity of the samples |
| `fd`, `precision`, `recall` | against the reference teacher at full steps |
| `data_fd`, `data_precision`, `data_recall` | against real data (when the reference config names a dataset) |
| `class_acc`, `colour_err`, `shape_err`, `center_tv` | the dataset's exact oracle (synthetic datasets) |
| `reward` | the reward from the reference config's `rl.reward` |
| `ms_per_sample` | timing; compare within a single run only |

Column names come from `packages/reward/src/diffusion_omni/reward/scorecard.py`. Read the table with the reference row at each step count in mind; a student that is better at one step count and worse at another is common.

Reference: [reward](../reference/reward.md). Toy results from the scorecard are in [docs/results/distillation_toy.md](../results/distillation_toy.md).
