# Scope: few-step distillation (`omnidiff-distill`)

Status: **v0 in progress**. Phases A and B are implemented; Phase C has started. Written 2026-10-17. The landscape claims below were checked against repos and docs on that date. Papers marked † were seen only as titles or abstracts and haven't been read in full.

## TL;DR

- **The gap is narrower than assumed for continuous image and video.** NVIDIA **FastGen** (Apache-2.0, Jan 2026, about 1k stars) is close to "Miles for distillation". It covers CM, sCM, TCM, MeanFlow, DMD2, f-Distill, LADD, CausVid, Self-Forcing and KD on EDM, DiT, SD, Flux, Qwen-Image, Wan, CogVideoX and Cosmos, with DDP and FSDP2. FastVideo and NVlabs/rcm ship production few-step video students. Another DMD2-for-Wan framework would be redundant.
- **Three gaps are still open and fit omnidiff well:**
  1. **Distilling discrete (text) diffusion.** No framework exists, only paper code (SDTT, DUO/DCD, and a burst of 2026 papers). omnidiff already has the masked process next to the continuous ones.
  2. **Distillation combined with RL.** FastGen has no RL. miles_diffusion has RL but explicitly no DMD2: its docs say teacher roles are "not … a new DMD2 loss", and the trainer has one trainable actor where DMD2 needs two. Nothing trains a few-step student and then RL-tunes it, or puts a reward inside the distillation loss. omnidiff's exact per-step log-probs make few-step students RL-able by construction.
  3. **A distillation scorecard.** The central failure mode is diversity and mode collapse (DMD2 at ≤2 steps, the reason rCM and 1.x-Distill exist). There is no standard teacher-relative report of quality, diversity, reward retention and NFE curve. This overlaps the reward/eval layer, so the work counts twice.
- **Recommendation:** build `omnidiff-distill` v0 around **one interface across text, image and video**. Ship KD, DMD2, consistency/flow-map, and discrete self-distillation as small, test-verified reference implementations. Spend research effort on gap 2 (RL ∘ distill) with gap 1 (text) as the differentiator. Phases A and B are CPU-only. Phase D (a real small model) needs a cost estimate and your sign-off first.

## 1. Method families

Each family below is described by what it trains against, what extra machinery it needs, and how it fails.

### 1a. Trajectory / regression distillation

The student regresses the teacher's multi-step ODE solution: knowledge distillation, progressive distillation (halving steps repeatedly), and rectified-flow reflow.

- Needs: teacher ODE solves, often precomputed noise→sample pairs.
- Fails: blurry outputs at 1–2 steps, because regressing to the conditional mean averages modes.
- 2026 state of the art: **PDD**, Parallel Decoding Distillation (NVIDIA, Jul 2026)†. It predicts several denoising steps per forward pass, avoids score-distillation and adversarial losses, reports strong 4–8 NFE results on LTX-2.3, Wan 14B and Qwen-Image, and is already on SGLang's Q4 serving roadmap. Also **TMD**, Transition Matching Distillation (2601.09881)†.

### 1b. Consistency models

Learn f(x_t, t) → x_0 that is constant along each probability-flow ODE trajectory.

- Variants: CM/CD (discrete-time, with a teacher ODE step for the target), iCT (training without a teacher), **sCM** (continuous-time TrigFlow parameterisation, which needs a JVP of the network), TCM (truncated, two-stage).
- **rCM** (ICLR 2026, NVIDIA/Tsinghua) adds a DMD-style reverse-divergence score-distillation regulariser to sCM. sCM's forward divergence loses fine detail; rCM fixes that. It roughly matches DMD2 quality with better diversity, at 1–4 steps, up to Wan 14B. Its main engineering contribution is a FlashAttention-2 JVP kernel that is compatible with parallelism. Code: NVlabs/rcm, which also has Causal-rCM.
- Needs: JVP (`torch.func.jvp`, which is fine at toy scale), an EMA or stop-grad target, careful time weighting.
- Fails: training instability (tangent normalisation and warmup matter), and quality that degrades as you add steps.

### 1c. Flow maps

Learn the two-time map X(x_t, t, s) that jumps from t to any s. Consistency models are the special case s = 0, and the instantaneous velocity is the limit s → t.

- Variants: Shortcut models, **MeanFlow** (average velocity, trains from scratch), **Align Your Flow** (NeurIPS 2025, continuous-time flow-map distillation; it proves that consistency models degrade as steps increase while flow maps do not; distils FLUX with LoRA), Mean Flow Distillation (2606.11155)†, and **Flow Map Distillation Without Data** (CVPR 2026, prior samples only: FID 1.45 one-step on ImageNet-256, which also sidesteps teacher/data mismatch).
- Needs: a two-time conditioning input (one more embedding), and JVPs for the continuous-time objectives.
- Why it matters here: one student serves any step count, which is the right shape for "serve at 1, 2, 4 or 8 steps depending on load".

### 1d. Distribution matching

Train the student so that its output distribution matches the teacher's, not its trajectories.

- **DMD2**: the gradient is the difference between the teacher's score and a *fake* score network trained online on student samples, using a two-time-scale update rule. A GAN head on the fake-score features replaces DMD's regression loss.
- Also: f-Distill (general f-divergences), LADD (latent adversarial), **1.x-Distill** (2604.04018† on fractional steps and diversity at ≤2 steps), **AdvDMD** (2604.28126†, which uses the DMD2 discriminator as a reward).
- Needs: **two trainable networks with separate optimisers and update ratios**, plus a frozen teacher. Three model copies in memory.
- Fails: mode collapse and diversity loss. The loss is mode-seeking by design.

### 1e. Causal / streaming video

CausVid and Self-Forcing distil a bidirectional teacher into an autoregressive few-step student, using DMD-style losses and training on the student's own rollouts. Causal-rCM is the consistency version. This is the bridge to the separate "causal video training" layer and is out of scope for v0.

### 1f. Discrete / language diffusion

This is the open frontier.

- **SDTT**: self-distillation through time for masked diffusion LMs. The student matches the teacher's logits after k teacher steps.
- **DUO + DCD**: discrete consistency distillation through the Gaussian↔uniform-state duality.
- A burst of 2026 work:
  - Multi-Mask DLMs (2607.19686)†: a single absorbing state leaves no entropy for consistency-style distillation, so it proposes multi-mask states with shared-Gumbel coupling.
  - Consistent Diffusion Language Models (ICML 2026)†: unifies diffusion, consistency and distillation objectives across masked and uniform corruption.
  - OPTD (2608.02942)†: on-policy transition distillation.
  - Trajectory self-distillation (2602.12262)† and temporal self-distillation (2609.15177)†.
  - Scaling Categorical Flow Maps (2605.07820)†: flow maps on the simplex.
  - Distilled continuous DLMs (2609.04531)†: code in one step.
  - "Simple DLMs are more effective few-step generators than reported" (2609.33947)†: a baselines warning to read **before** building anything.
- Note on dLLM "parallel decoding" tricks such as confidence remasking: these are inference heuristics, not distillation. omnidiff can host them as inference-only samplers.

## 2. Who covers what (2026-10-17)

| | continuous methods | discrete/text | RL in the loop | eval of students | serving handoff | scale |
|---|---|---|---|---|---|---|
| **NVIDIA FastGen** | CM, sCM, TCM, MeanFlow, DMD2, f-Distill, LADD, CausVid, Self-Forcing, KD | none | none | scripts only | `.pth` checkpoints | ≥10B, FSDP2 |
| **FastVideo** | DMD2, sparse distillation (VSA), Self-Forcing | none | none | none | ships HF models; SGLang Diffusion forked from it | Wan 14B, MiniMax-H3 |
| **NVlabs/rcm** | rCM, Causal-rCM | none | none | some | checkpoints | Wan 14B |
| **miles_diffusion** | teacher/ref model roles, custom-loss hooks, **no DMD2** (one trainable actor) | none | Flow-GRPO, DiffusionNFT, SFT | built-in scorers | rollouts served by SGLang | FSDP2 |
| **SGLang Diffusion** | serves distilled models; Q4-2026 roadmap: PDD, Turbo LoRA on FastH3 V2, distilled LoRAs that survive merging | none | none | quality gate (SSIM/PSNR vs BF16) | is the server | production |
| **omnidiff (today)** | none | masked process exists | GRPO/DDPO with exact log-probs for all three modalities | none | none | toy, single process |

Sources: [FastGen](https://github.com/NVlabs/FastGen), [FastVideo](https://github.com/hao-ai-lab/FastVideo), [NVlabs/rcm](https://github.com/NVlabs/rcm), [miles_diffusion model roles](https://github.com/radixark/miles_diffusion/blob/main/docs/fsdp_model_roles.md), [SGLang-Diffusion roadmap 2026 Q4](https://github.com/sgl-project/sglang/issues/43313).

## 3. Gaps, ranked by fit to this repo and budget

| # | gap | open? | fit | compute |
|---|---|---|---|---|
| G1 | Text/discrete diffusion distillation in a framework, sharing an interface with image/video | Yes, only paper code | High: the masked process already exists | Low at toy scale, moderate for a 100M+ dLLM |
| G2 | RL ∘ distillation: RL-tune few-step students, add reward terms to distillation losses, compare "distil then RL" with "RL then distil" | Yes: FastGen has no RL, miles has no DMD2 | High: exact few-step log-probs come from `transition` | Low to moderate |
| G3 | Distillation scorecard: teacher-relative quality, diversity/coverage, reward retention, quality-vs-NFE curve | Mostly | High: shared with the reward/eval layer | Low |
| G4 | Serving handoff: export student + sampler spec (timesteps, CFG baked in, LoRA) that diffusers/SGLang load | Partly; SGLang is building the serving half | Medium: a valuable upstream PR later | Low |
| G5 | Large-scale continuous distillation (DMD2/rCM on Wan/Flux) | **No**: FastGen, FastVideo and rcm cover it | Low | Very high, skip |

## 4. Proposed v0 scope

### Core changes (in `omnidiff-core`)

1. **Arbitrary timestep sets and one-jump samplers.** `sample(..., timesteps=[1.0, 0.5, 0.0])`, and a `FewStepSampler` that calls a student map directly, f(x_t, t[, s]) → x_s.
2. **Prediction conversions on each process**: `to_x0(pred, x_t, t)`, `to_score(...)`, and `ode_step(pred, x_t, t, s)` for the deterministic teacher step. DMD needs scores, CD needs teacher ODE steps, and SDTT needs teacher logits after k steps.
3. **Two-time conditioning** (an `s` embedding) in `Denoiser`, optional and off by default, for flow maps.

### The `omnidiff-distill` package

- A `Distiller` interface: `roles` (student, teacher, plus auxiliaries such as fake_score and discriminator), one optimiser and update ratio per trainable role, `loss(batch) -> {role: loss}`, and the `student_sampler()` to export.
- A `DistillTrainer` with multiple trainable roles, two-time-scale updates, EMA, and a frozen teacher (LoRA-disabled base or a separate copy, same as the RL trainer). This multi-role trainer is the structural thing miles_diffusion lacks.
- Methods, each a registered `DISTILLERS` entry with a toy config for text, image and video where the method applies:

| method | modalities | why in v0 |
|---|---|---|
| `kd` (trajectory regression, progressive) | image, video, text (logit KD) | baseline everything else must beat |
| `dmd2` | image, video | the industry default; exercises the multi-role trainer |
| `consistency` (CD first, then sCM via `torch.func.jvp`) | image, video | the rCM path; JVP correctness tested against finite differences |
| `flow_map` (MeanFlow / AYF-style) | image, video | any-step student; the most promising direction |
| `sdtt` (self-distillation through time) | text | G1 entry point, simple, and a strong published baseline |
| `dcd` or multi-mask consistency | text | G1 research, **only after** re-reading 2609.33947 and 2607.19686 |

- A scorecard (`omnidiff-reward` side, G3), computed on the same prompts for teacher@N and student@{1, 2, 4, 8}:
  - Teacher-relative quality: a toy Fréchet distance on a small fixed feature net for pixels, teacher NLL for text.
  - Diversity: pairwise distance within prompt, and mode coverage on the toy datasets where the modes are known (the blob dataset has exact colour/position modes).
  - Reward retention.
  - Wall-clock time and NFE.

### The research thread (G2)

- RL-tune a 2–4-step student with GRPO. The flow SDE with 2–4 steps is a high-variance policy, so measure that.
- Reward-regularised DMD2: add a reward gradient or advantage-weighted term to the generator loss.
- Order effects: SFT → RL → distil versus SFT → distil → RL, measuring reward and diversity.
- Hypothesis to test: distillation's mode-seeking compounds RL's mode collapse. The scorecard should show it.

### Out of v0

Models of 1B+ parameters, FSDP, sparse attention, causal/streaming students (Self-Forcing / Causal-rCM), quantisation-aware distillation, and pretrained-weight loaders.

## 5. Phases and exit criteria

| phase | work | exit criteria | compute |
|---|---|---|---|
| A | core changes (§4 items 1–3); multi-role `DistillTrainer`; `kd`, `dmd2`, `sdtt`; scorecard v0 | on toy blobs, a 1–2-step student within a set tolerance of teacher@20 on Fréchet distance; on toy text, a 4-step SDTT student beats the 4-step undistilled teacher; all methods pass CPU tests | laptop CPU |
| B | `consistency` (CD → sCM/JVP), `flow_map` | JVP test against finite differences; flow-map student monotone or flat in quality as steps increase (the AYF claim) on toys | laptop CPU |
| C | G2 experiments | a written result, positive or negative, on order effects and reward-regularised DMD2 | laptop CPU, maybe 1 small GPU |
| D | one real small model per modality (candidates: MDLM-OWT-class dLLM, ImageNet-64 EDM/EDM2) | reproduce one published number within noise before claiming anything new | **GPU: estimate and ask first** |
| E | G4 export to the diffusers/SGLang format, and an upstream issue or PR | a student checkpoint served by SGLang Diffusion with its sampler spec | small |

### Reading notes: 2609.33947 (abstract only)

- Much of the few-step quality gap in masked DLMs is a sampler artifact. "Modest sampler sharpening", meaning temperature below 1 with no retraining, makes an old MDLM-class model rival newer ones. At 16 steps it beats the default sampler at 1024 steps on generative perplexity.
- Temperature 1 is generically suboptimal under parallel unmasking, because tokens decoded together lose their dependencies.
- Per-output metrics can be gamed by a generator with two outputs, so quality and across-output diversity must be scored jointly. The paper calls this GroupEval.
- What this means here: every text distillation result is compared with a temperature-tuned teacher at the same step count, and reported with coverage and diversity next to validity. The scorecard does this through `variants`.

## 6. Risks

- **Crowding.** FastGen could add discrete or RL support. Mitigation: check its issues and roadmap before Phase C, and keep the differentiator (text, RL, scorecard) at the centre.
- **Discrete distillation is moving monthly.** There were five or more relevant arXiv papers between Jun and Sep 2026. Read 2609.33947 ("more effective than reported") first; the field's baselines may be wrong.
- **Toy-to-real transfer.** Blob and toy-text results won't transfer automatically. Phase D exists to anchor one result at real scale.
- **Budget.** Anything beyond Phase C needs GPU time. Each GPU run gets a written cost estimate beforehand.

## 7. Decisions (2026-10-17)

1. **Primary differentiator.** G2 (RL ∘ distillation) is the research thread and G1 (text distillation) is the visible feature.
2. **Re-implement in pure PyTorch.** There is no FastGen dependency. Reference behaviour is pinned by tests instead.
3. **Phase D is deferred.** The work stays on CPU toys until there is a cost estimate and a compute source.

## 8. v0 implementation status

| item | status | where |
|---|---|---|
| process helpers (`q_sample`, `to_x0`, `to_eps`, `ode_step`), explicit grids | done | `omnidiff.core.processes`, `sampling.resolve_timesteps` |
| renoise sampler as an exact Gaussian policy | done | `ContinuousProcess.renoise_transition` |
| two-time models, math attention for JVP | done | `Denoiser(two_time=True)`, `layers.math_attention` |
| multi-role trainer, teacher pool, EMA, export with sampler spec | done | `omnidiff.distill.trainer` |
| `kd`, `dmd2` (+ GAN head), `consistency`, `flow_map`, `sdtt` | done | `omnidiff.distill.methods` |
| scorecard: FD / precision / recall vs teacher and data, diversity, oracles, reward, latency | done | `omnidiff.reward.scorecard`, dataset `oracle()` |
| RL on few-step students | done; order study in results | `test_renoise_student_is_rl_tunable` |
| toy results | done (single seed): see [docs/results/distillation_toy.md](../results/distillation_toy.md) | |
| noise-conditioned text distillation (`coupled_kd`) | done; 4-step validity 0.66 against 0.20 for the tuned teacher on the toy | `methods/coupled_kd.py` |
| tuned-baseline rows in the scorecard (teacher + sampler/temperature overrides) | done | `run_scorecard(variants=...)`, `--ref-variant` |
| discrete consistency (DCD / multi-mask) | not started; read 2607.19686 first | |
| rCM (flow map with `s_mode: zero` + DMD regulariser) | not started; needs composable objectives | |
| progressive distillation rounds for continuous processes | not started | |

## References

- Zheng et al., *Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency* (rCM), ICLR 2026. [arXiv 2510.08431](https://arxiv.org/abs/2510.08431), [code](https://github.com/NVlabs/rcm)
- Sabour et al., *Align Your Flow: Scaling Continuous-Time Flow Map Distillation*, NeurIPS 2025. [arXiv 2506.14603](https://arxiv.org/abs/2506.14603)
- *Flow Map Distillation Without Data*, CVPR 2026. [paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Tong_Flow_Map_Distillation_Without_Data_CVPR_2026_paper.pdf)
- *Mean Flow Distillation*†, [arXiv 2606.11155](https://arxiv.org/abs/2606.11155)
- Shaul et al., *Parallel Decoding Distillation for Fast Image and Video Generation* (PDD)†, [arXiv 2607.26004](https://arxiv.org/abs/2607.26004), [project](https://research.nvidia.com/labs/genair/pdd/)
- *Transition Matching Distillation for Fast Video Generation*†, [arXiv 2601.09881](https://arxiv.org/abs/2601.09881)
- Yin et al., *Improved Distribution Matching Distillation* (DMD2), [arXiv 2405.14867](https://arxiv.org/abs/2405.14867)
- *1.x-Distill*†, [arXiv 2604.04018](https://arxiv.org/abs/2604.04018); *AdvDMD*†, [arXiv 2604.28126](https://arxiv.org/abs/2604.28126)
- Lu & Song, *Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models* (sCM), [arXiv 2410.11081](https://arxiv.org/abs/2410.11081)
- Geng et al., *Mean Flows for One-step Generative Modeling*, [arXiv 2505.13447](https://arxiv.org/abs/2505.13447)
- Deschenaux & Gulcehre, *Beyond Autoregression: Fast LLMs via Self-Distillation Through Time* (SDTT), [arXiv 2410.21035](https://arxiv.org/abs/2410.21035)
- Sahoo et al., *The Diffusion Duality* (DUO, DCD), [arXiv 2506.10892](https://arxiv.org/abs/2506.10892)
- 2026 discrete few-step work†: [2607.19686](https://arxiv.org/abs/2607.19686), [2608.02942](https://arxiv.org/abs/2608.02942), [2602.12262](https://arxiv.org/abs/2602.12262), [2609.15177](https://arxiv.org/abs/2609.15177), [2605.07820](https://arxiv.org/abs/2605.07820), [2609.04531](https://arxiv.org/abs/2609.04531), [2609.33947](https://arxiv.org/abs/2609.33947)
- Frameworks: [NVlabs/FastGen](https://github.com/NVlabs/FastGen), [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo), [radixark/miles_diffusion](https://github.com/radixark/miles_diffusion), [SGLang Diffusion](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen)
