# Tutorials

Six tutorials on diffusion post-training and few-step distillation, using this repo's toy tasks. They assume you know PyTorch. They do not assume you know diffusion distillation or RL for diffusion.

Everything here runs on CPU in seconds to a couple of minutes. The numbers you get from the tutorial commands are tiny-model sanity numbers. They show that a pipeline works and what a symptom looks like. They are not the results of the project. Results are in [docs/results](../results/), and every tutorial links there when it quotes one.

Scope: only merged features (core, posttrain, distill, scorecard). The reward service, the miles adapter and the `eval` command are coming and are not covered.

## Suggested order

| # | Tutorial | You learn | Needs |
|---|---|---|---|
| 1 | [Denoising as a policy](01-denoising-as-a-policy.md) | processes, `transition`, exact log-probs for flow, Gaussian and masked diffusion | nothing |
| 2 | [From SFT to RL](02-sft-to-rl.md) | the SFT loss, GRPO on the denoising MDP, LoRA and the KL reference | the teacher from tutorial 1 |
| 3 | [Few-step distillation](03-few-step-distillation.md) | KD, consistency, DMD2, flow map, progressive, and why a tuned teacher is the baseline | the teacher from tutorial 1 |
| 4 | [Text diffusion and coupled distillation](04-text-and-coupled-distillation.md) | why few-step masked decoding fails, `coupled_kd`, SDTT, `dcd` | its own text teacher |
| 5 | [Rewards, reliability and hacking](05-rewards-reliability-hacking.md) | reward API, KL anchors, reward-weighted distillation, seeds and error bars | the teacher from tutorial 1 |
| 6 | [Export and eval](06-export-and-eval.md) | scorecard anatomy, export bundles, the multi-seed report | students from tutorial 3 and 4 |

Read 1, 2, 3 in order. 4 and 5 can be read in either order after 3. 6 last.

## One-time setup

Run from the repository root, in one shell session (the tutorials use shell variables and arrays, which work in bash and zsh).

```bash
export OUT="${TMPDIR:-/tmp}/diffusion-omni-tutorials"   # any writable directory
mkdir -p "$OUT"
# A tiny image model and a tiny text model. Appended after -c <config>, these override the shipped config.
TINY_IMG=(device=cpu model.dim=32 model.backbone.depth=1 model.backbone.heads=2 sampling.num_steps=4)
TINY_TXT=(device=cpu model.dim=64 model.backbone.depth=2 model.backbone.heads=2 sampling.num_steps=16)
```

Every command ends with `output_dir=$OUT/...`, so nothing is written under `runs/`.

Two conventions to know:

- A config is YAML. Anything after `-c <config>` of the form `a.b.c=value` overrides the config (`diffusion_omni.core.config`). The values are parsed as YAML, so `[1, 2]`, `null` and `1e-4` work. The tiny overrides above are the same idea as `diffusion_omni.core.testing.tiny_config`, which the test suite uses.
- A checkpoint is a directory with `model.pt` and `config.yaml`. The trainers write `<output_dir>/sft/final`, `<output_dir>/rl/final` and `<output_dir>/distill/final`, plus `log.jsonl` next to them.

Where things live:

| Topic | Code |
|---|---|
| processes | `packages/core/src/diffusion_omni/core/processes/` |
| sampling loop | `packages/core/src/diffusion_omni/core/sampling.py` |
| SFT, RL | `packages/posttrain/src/diffusion_omni/posttrain/` |
| distillers | `packages/distill/src/diffusion_omni/distill/methods/` |
| rewards, metrics, scorecard | `packages/reward/src/diffusion_omni/reward/` |

Background for the whole set: [docs/ARCHITECTURE.md](../ARCHITECTURE.md), [docs/results/distillation_toy.md](../results/distillation_toy.md), [packages/distill/README.md](../../packages/distill/README.md).

## How the commands were checked

Each fenced `bash` block in the tutorials was run from a clean `$OUT` on CPU. The output shown under a command is from that run. Your numbers will differ in the last digits across machines. Statements that say "equal" or "zero" hold exactly.
