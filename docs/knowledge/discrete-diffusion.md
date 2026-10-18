# Discrete / masked diffusion and few-step discrete methods

Date read: 2026-10-18. All entries are **abstract-only reads** of the arXiv abs page. Scope context: [docs/scope/distillation.md](../scope/distillation.md) section 1f.

## Base models

- **MDLM** (Simple and Effective Masked Diffusion LMs), Sahoo et al., 2024-06-11, NeurIPS 2024. [2406.07524](https://arxiv.org/abs/2406.07524). Rao-Blackwellised objective that is a mixture of masked-LM losses; says masked diffusion is "more performant than previously thought"; new SOTA among diffusion LMs, approaches AR perplexity (no numbers in abstract).
- **LLaDA**, Nie et al., 2025-02-14. [2502.09992](https://arxiv.org/abs/2502.09992). 8B masked diffusion LM trained from scratch (pretrain + SFT); described as competitive with LLaMA3 8B in in-context learning; no benchmark numbers in abstract.

## Distillation and few-step methods

- **SDTT**, Deschenaux and Gulcehre, 2024-10-28. [2410.21035](https://arxiv.org/abs/2410.21035). Self-distillation through time cuts inference steps 32-64x; up to 860M parameters demonstrated; at 1.3B reports up to 8x faster than AR with KV cache.
- **DUO / Discrete Consistency Distillation**, Sahoo et al., 2025-06-12, ICML 2025. [2506.10892](https://arxiv.org/abs/2506.10892). Uniform-state diffusion derived from Gaussian diffusion; curriculum "doubling training speed"; DCD gives about 100x faster sampling. Beats AR on zero-shot perplexity on 3 of 7 benchmarks.
- **Multi-Mask DLMs (MultiMDM)**, Chen et al., v1 2026-07-22, v3 2026-09-02 (v2 withdrawn), COLM 2026. [2607.19686](https://arxiv.org/abs/2607.19686). States all MDM forward trajectories collapse to one fully masked state, leaving no terminal entropy for consistency-style distillation; keeps masking but mixes over a mask set; closed-form ELBO; discrete-state consistency distillation with shared-Gumbel coupling. No numbers in abstract.
- **OPTD**, Lu et al., 2026-08-03. [2608.02942](https://arxiv.org/abs/2608.02942). On-policy transition distillation choosing the longest verified prefix of future commitments; reports strongest quality-constrained AUP on four math/code benchmarks; no values in abstract.
- **Trajectory self-distillation**, Zhang et al., v1 2026-02-12, v4 2026-07-03. [2602.12262](https://arxiv.org/abs/2602.12262). Student matches full-step teacher trajectory; adds Direct Discriminative Optimization (reverse KL, mode-seeking). No numbers in abstract.
- **Temporal Self-Distillation**, Xu et al., 2026-09-14. [2609.15177](https://arxiv.org/abs/2609.15177). Single-stage on-policy; earlier-timestep predictions match the distribution at the commit timestep; seven benchmarks (math, planning, code). No numbers in abstract.
- **Scaling Categorical Flow Maps**, Davis et al., 2026-05-08. [2605.07820](https://arxiv.org/abs/2605.07820). 1.7B base trained on 2.1T tokens, self-distilled to a categorical flow map, diverse text in as few as 4 steps.
- **Distilled Continuous DLMs (PlaidQ)**, Peng et al., v1 2026-09-03. [2609.04531](https://arxiv.org/abs/2609.04531). **The arXiv page shows v2 WITHDRAWN** (2026-09-10) over institutional disclosure requirements. Its v1 claims (16-step student 31.78 pass@10 HumanEval, above the 512-step teacher) should not be relied on.
- **Consistent Diffusion Language Models** (ICML 2026 poster): not fetched. Unverified.

## The sampler-tuning paper (2609.33947)

"Simple Diffusion Language Models Are More Effective Few-Step Generators Than Reported", Amin, Yin, Khanna, 2026-09-27, v1. [abs](https://arxiv.org/abs/2609.33947). Abstract read. States:
- Much of the few-step gap is a sampler configuration issue; mild sharpening, no retraining, lets an older masked DLM rival newer ones.
- The sharpened sampler reaches lower generative perplexity at 16 steps than the standard sampler at 1024.
- Any optimal trade-off between two per-output metrics can be reached by a generator supported on at most two outputs; GroupEval scores quality and across-output semantic diversity separately.
- A distilled model's "1.5-4.7x perplexity gains" yield no matching quality gain under GroupEval.
- Theory: parallel unmasking breaks dependencies among simultaneously decoded tokens, so temperature one is generally suboptimal even for an exact denoiser.
Not stated in the abstract: that the sharpening is specifically "temperature below 1" (abstract says "sharpening"); docs/scope assumes this reading.

## Relevance to this repo

- Masked process: `packages/core/src/diffusion_omni/core/processes/masked.py`.
- Text distillation: `packages/distill/src/diffusion_omni/distill/methods/sdtt.py`, `coupled_kd.py`, `dcd.py`.
- Tuned-teacher baseline rows (the 2609.33947 lesson): `run_scorecard(variants=...)` in `packages/reward/src/diffusion_omni/reward/scorecard.py`.
- Within-prompt diversity metric aligned with GroupEval's idea: `within_prompt_diversity` in `packages/reward/src/diffusion_omni/reward/metrics.py`. It is not GroupEval (not read beyond the abstract).
