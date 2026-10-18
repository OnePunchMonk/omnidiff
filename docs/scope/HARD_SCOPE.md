# Hard scope

Status: proposal, 2026-10-18. This is the definition of "stable" for diffusion-omni. Anything not listed under In is not part of the stable cut.

## 1. One-sentence claim

diffusion-omni is the only place where text (masked), image (Gaussian) and video (flow) diffusion share one policy interface with exact per-step log-probabilities, and where distillation, RL and reward reliability are studied on that interface with reproducible, multi-seed evidence.

## 2. Position against neighbours

| project | owns | what diffusion-omni does instead |
|---|---|---|
| diffusers | pretrained pipelines, schedulers, LoRA, training scripts for image/video/audio | no model zoo. Import from it when a real model is needed. Contribute the policy interface idea only if it proves out |
| SGLang Diffusion | serving: kernels, caching, quantisation, parallelism | no serving work. Produce students it can load |
| miles-diffusion | RL/SFT at scale (FSDP2, Flow-GRPO, DiffusionNFT) | research bed at toy scale, text diffusion included; offer a reward backend it can call |
| FastGen, FastVideo, rCM | continuous few-step distillation at scale | do not compete there. Work on the cases they skip (see claim 2) |
| ComfyUI, mflux, DiffusionKit | workflows, edge | out of scope |

## 2a. Borrow, do not rewrite

The intent is to borrow code from upstream repos as much as possible. diffusion-omni's own code is the part that is the gap (the policy interface, text-diffusion distillation, reward reliability, the scorecard). Everything else should come from upstream:

- **Dependency first.** Prefer importing a library (diffusers schedulers and models, SGLang, FastGen, FastVideo, miles-diffusion) as an optional extra in the package that needs it.
- **Vendor second.** Where importing is impractical, copy the needed code into the package with the licence header, the upstream URL and commit, and a note of what changed. Keep vendored files in a `_vendor/` or clearly marked module so they can be refreshed.
- **Write third.** Write new code only for the four claims, or when no upstream exists. Say in the scope doc which upstream was checked.
- **Keep core small.** `diffusion-omni-core` stays torch + numpy + pyyaml; upstream code goes into other packages or optional extras, not into core.
- **Attribute and contribute back.** List borrowed code in `docs/knowledge/frameworks.md` with licence and commit. When we fix or extend something we borrowed, send it upstream where the maintainers want it.
- **Licence check before copying.** Compatible licences only (Apache-2.0, MIT, BSD); record it in the PR.

## 3. The four claims

### Claim 1: one policy interface with exact log-probs across masked, Gaussian and flow

- **Problem.** RL on diffusion needs the log-probability of each reverse step. Frameworks implement it per model family; text diffusion usually has none.
- **Deliverable.** `DiffusionProcess` in core: `forward`, `reverse_step`, `log_prob`, `sample_step`, for three processes, behind one model contract (adapter, backbone, adapter).
- **Gate (tests, CPU).**
  - on-policy ratio is 1 to 1e-6 for every process;
  - log-prob equals the closed form on a Gaussian toy and an enumerable masked toy;
  - the same GRPO and DDPO code runs unchanged on all three modalities and reward rises on each.
- **Already done.** v0.1 in core and posttrain; 165 tests.
- **Remaining.** Freeze and document the contract; config validation; resume exactness for RL.
- **Non-goals.** New processes beyond the three; real text encoders.

### Claim 2: few-step distillation for discrete diffusion, and RL composed with distillation

- **Problem.** Continuous distillation is covered at scale. Few-step text diffusion is open: independent per-position sampling breaks at low NFE. And no one reports how RL and distillation should be ordered.
- **Deliverables.**
  - text students (`sdtt`, `coupled_kd`) with a scorecard of validity, coverage and diversity against NFE;
  - the order study: distill then RL, RL then distill, with and without a KL anchor, on image, video and text;
  - a decision on the default teacher time embedding for new configs.
- **Gate.**
  - 3-seed tables with confidence intervals on the default and the `time_scale: 10` teachers;
  - each open issue in `docs/results/distillation_toy.md` is closed or written up as a negative result;
  - `coupled_kd` beats `kd` and `sdtt` at 1 and 2 steps outside CI overlap, or the claim is withdrawn in the doc.
- **Already done.** Ten methods merged; single-seed headline results; export bundles; resume.
- **Remaining.** E1 to E4 runs; RL on students (M2); `coupled_kd` with a real tokenizer.
- **Non-goals.** Beating FastGen on continuous images; new distillation methods without a failing case that motivates them.

### Claim 3: reward reliability as a measured property

- **Problem.** RL results are only as good as the reward. Rewards get hacked and no tool measures it before training.
- **Deliverable.** A reliability study API: agreement (Spearman with bootstrap CI) between each reward and each oracle; hacking probes (RL on reward A, track held-out reward B and the oracle); a pass/fail report per reward.
- **Gate.**
  - tests: perfect agreement gives 1, ties are handled, inverse ranking gives -1, CIs shrink with n;
  - a written study on at least two modalities, with one reward shown to be hackable and one shown to be robust, or a negative result stated plainly.
- **Already done.** Toy rewards and metrics. R1 to R4 coded, in review.
- **Remaining.** Merge R1 to R4 with review findings fixed (service override must accept registered names only); run R3; throughput measurement.
- **Non-goals.** VLM judges (optional extra later); building a reward-model zoo.

### Claim 4: one student scorecard

- **Problem.** Each distillation paper reports a different metric at a different NFE. Students cannot be compared.
- **Deliverable.** `diffusion-omni scorecard` and `eval`: teacher versus student across NFE with set metrics (FD, precision, recall, diversity), exact oracles, tuned reference samplers, seeds and confidence intervals. One versioned JSON with provenance (config hash, seed, git sha) and a docs table generated from it.
- **Gate.**
  - metric identities hold (FD of a set with itself is 0);
  - no NaN in output, duplicate seeds rejected;
  - rerunning produces the same JSON;
  - a regression check compares a run to a stored baseline.
- **Already done.** Scorecard v0 and the multi-seed harness.
- **Remaining.** Fold the quality-vs-NFE Pareto table in; baseline regression check; schema `diffusion_omni.eval/v1`.
- **Non-goals.** Published-benchmark adapters (GenEval, VBench) until a real model exists.

## 4. In the stable cut

| package | role |
|---|---|
| core | claim 1 |
| posttrain | claim 1, claim 2 (RL side) |
| distill | claim 2 |
| reward | claim 3 |
| eval (initially inside reward) | claim 4 |
| reference samplers (initially inside core or eval) | ground truth the scorecard compares against; slow, exact, float64 option |

## 5. Out or deferred

| item | why |
|---|---|
| optim (caching, quantisation, compile) | SGLang, diffusers and research code own it |
| edge | mflux, DiffusionKit, ExecuTorch own it; revisit for few-step students after claim 2 |
| workflow runtime | ComfyUI owns it |
| safety, watermark | not a diffusion-training gap |
| multi-tenant LoRA | overlaps SGLang serving; upstream instead |
| causal video | real gap but needs real video compute; after M4 |
| data engine | NeMo curator and img2dataset own it; keep the toy datasets |
| bench (performance) | fold the NFE-quality table into claim 4; no separate package |
| RSI-inference (agents improving diffusion inference engines) | separate research thread, see section 9; not part of the stable cut |

## 6. Kill criteria

- If `coupled_kd` does not beat the baselines at low NFE on 3 seeds, claim 2's text half is withdrawn and the README says so.
- If no reward is found hackable and none robust, claim 3 is reported as a negative result, not dropped silently.
- If claim 1 cannot hold on any of the three processes, the repo is reduced to the processes where it does.

## 7. Stable means

1. The four gates above pass on `main`.
2. `uv sync && uv run pytest -q` and `scripts/check_docs.py` are green on a fresh clone, enforced by CI.
3. Every number in the docs comes from a multi-seed run with raw JSON committed next to it.
4. No benchmark-aspiration wording.
5. One real-scale result (M4: one small real model, one published number reproduced within noise) shows claim 2 or 3 holds outside toys. This needs a written cost estimate and sign-off first. Without it the repo is a research toolkit and the README says so.

## 8. Order of work

1. Merge R1 to R4 with fixes (claim 3, 4).
2. Runs E1 to E5 (claim 2, 3).
3. Fold eval and reference samplers (claim 4).
4. Docs rewrite and CI.
5. Phase D cost estimate, then sign-off or defer.

## 9. Separate thread: RSI-inference for diffusion

Source: rsi-inference.com redirects to Apple's `ml-rsi-inference-bench`. It is a benchmark where a coding agent works up to 8 hours on a pinned vLLM or SGLang engine with 1 to 8 GPUs, and a separate verifier builds, serves and measures the result. Tracks: Day-0 model support (pass@1), inference optimisation and regression fixes (geometric-mean speed-up, with a failed correctness check scored 1.00x). 37 tasks, of which only 3 are diffusion (1 continuous, 2 discrete). This replaces my earlier reading of "rsi" as a self-training loop; that reading was wrong.

Why it is a real gap: diffusion inference is barely covered, and a verifier needs a ground truth for "correct". Our exact reference samplers (claim 4) are that ground truth: a candidate optimisation passes only if its samples match the reference within a stated tolerance or distribution test.

Possible deliverable (not started, needs a decision): a diffusion track for this style of benchmark with SGLang Diffusion tasks (day-0 support for a new diffusion model, step-cache or quantisation speed-ups, regression fixes), verified by reference-sampler parity.

Constraints: it needs real GPUs and a pinned SGLang Diffusion commit, so it is blocked on the compute budget and cost sign-off. Toy-scale part that can be done now on CPU: the verifier (reference-parity and speed measurement on toy processes). Do not build agent scaffolding; use existing agent harnesses.
