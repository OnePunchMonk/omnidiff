# Build plan to Saturday 2026-10-31

Tracking: [goals and plan, #2](https://github.com/OnePunchMonk/diffusion-omni/issues/2) and [experiment runs, #3](https://github.com/OnePunchMonk/diffusion-omni/issues/3). How agents run, review and merge: [docs/plan/EXECUTION.md](docs/plan/EXECUTION.md). Runs and CPU-hour estimates: [docs/plan/BENCHMARK_RUNS.md](docs/plan/BENCHMARK_RUNS.md). Proposed rename to `diffusion-omni`: [docs/plan/RENAME.md](docs/plan/RENAME.md).

The previous plan (distillation, then the reward harness, by Fri 23) is done for distillation and half done for reward. This plan covers what is left and every layer in the [layer map](docs/GOAL_AND_PLAN.md#layer-map), so that by the 31st each layer is either a working, tested v0 package or explicitly deferred with a reason.

## Where things stand (2026-10-18)

- `main` at `9fe1202`, pushed, 165 tests passing (`uv run pytest -q`, about 7 s on CPU).
- Distillation merged: `kd`, `dmd2` (+ `last_stage_prob`), `dmd2_reward`, `consistency`, `flow_map`, `rcm`, `progressive`, `sdtt`, `coupled_kd`, `dcd` (single-mask ablation, negative); `diffusion-omni export` bundles; the multi-seed harness `scripts/distill_toy_report.py` with provenance; exact save and resume (D7); `docs/ARCHITECTURE.md`; a config for every method on every process it supports.
- Reward wave, not merged, open findings from the Codex review:
  - `ws/R1` (`6d019c8`), batched reward service: remote code execution through `--allow-override`, because an override `reward` spec can name any `module:object`; request validation gaps; a flaky timing test.
  - `ws/R2` (`841c63e`), miles-diffusion adapter: a single-frame video is classified as an image; bfloat16 outputs fail; an empty prompt becomes `None`.
  - `ws/R4` (`c0c8094`), `diffusion-omni eval`: NaN is written into the JSON; duplicate seeds and invalid `-n` / `--nfe` are accepted.
  - `ws/R3`, reliability study: no commits. Its worktree (`.claude/worktrees/agent-a4d78f7d54c6f83fe`) holds uncommitted `reliability.py`, `scripts/reward_reliability.py`, `test_reliability.py`, and an edit to `packages/data/.../datasets.py`, which R3 does not own.
- The `time_scale: 10` video teacher is trained: `runs/video_flow_ts10/sft/final` (8000 steps, 481 s, 2026-10-18 00:17). It has not been scored yet.
- Not run: the 3-seed distillation report, the ts10 3-seed report, the D2 advantage-mode run, the order study on the ts10 teacher, R3 and R1 measurements. A 3-seed report was started and stopped at `image_flow_consistency_s0`. Its four `_s*` run dirs predate the D7 trainer change, so the harness will likely flag them as stale.
- README is stale: it omits `progressive`, `dcd`, `dmd2_reward`, `export` and `ARCHITECTURE.md`, and points to `tests/test_e2e.py`, which is now `packages/posttrain/tests/test_posttrain_e2e.py`.

## What "the entire monorepo done" means on the 31st

Each layer below either meets its exit criterion on `main`, with `uv run pytest -q` green, or is recorded as deferred in `docs/GOAL_AND_PLAN.md` with the reason. "v0" means one capability that works end to end on CPU toys, with a test of the property that makes it correct. It is not feature parity with the frameworks named in the layer map.

| layer | package | done on the 31st when | status target |
|---|---|---|---|
| core | `diffusion-omni-core` | unchanged contract; any change made for a layer below (for example a causal attention mask) has its own test; README and ARCHITECTURE match the code | done |
| post-training | `diffusion-omni-posttrain` | unchanged; README toy table re-checked | done |
| distillation | `diffusion-omni-distill` | 3-seed tables for the default and ts10 teachers in `docs/results/distillation_toy.md`; scope §8 has no "not started" rows; open issues closed or written up as negative results | done (M1, M2) |
| reward and eval | `diffusion-omni-reward` | R1–R4 merged with all review findings fixed; R3 toy result written; R1 throughput measured | done (M3) |
| data engine | `diffusion-omni-data` | v0: manifest of shards; exact and near-duplicate removal; reward or oracle threshold filter; codec precompute into shards; `diffusion-omni data build` | v0 |
| causal / streaming video | new `diffusion-omni-causal` | v0: frame-causal attention for `video_patch`; per-frame noise levels in training (diffusion-forcing style); rolling-window generation past the trained length on `moving_blob` | v0, highest risk |
| multi-tenant LoRA | new `diffusion-omni-lora` | v0: one base model, N LoRA adapters, per-sample adapter index in one batched forward | v0 |
| edge deployment | new `diffusion-omni-edge` | v0: export bundle to a `torch.export` program of the unrolled few-step sampler; parity test against eager | v0 |
| workflow runtime | new `diffusion-omni-workflow` | v0: a small JSON/YAML graph of diffusion-omni nodes (prompts, sample, decode, reward, select) run by a topological executor, with a printed plan | v0 |
| safety and provenance | new `diffusion-omni-safety` | v0: a provenance manifest per generation (weights hash, config, sampler, seed); a keyed spread-spectrum watermark for pixel outputs with a detector | v0 |
| real-scale anchor (M4, Phase D) | n/a | a written cost estimate in `docs/plan/PHASE_D_COST.md` for the user to sign off. No GPU run | deferred: needs sign-off and compute |
| handoff (M5): diffusers / SGLang | n/a | not attempted: serving a toy DiT in SGLang proves nothing, and a real student needs Phase D | deferred until after M4 |
| pretraining, VAE / tokenizer training | n/a | stays out of scope | out of scope |

### v0 exit criteria for the new layers (all CPU-testable)

| id | layer | exit criteria (tests) | deferred inside the layer |
|---|---|---|---|
| L1 | data engine | an injected exact duplicate and a near-duplicate (random-projection features, cosine above a threshold) are removed and nothing else is; the filter keeps exactly the samples at or above the threshold; precompute then reload reproduces `codec.encode` bit for bit; a manifest built twice from the same inputs is byte-identical | recaptioning (needs a VLM), web-scale sources, a real VAE |
| L2 | multi-tenant LoRA | a mixed batch over N adapters equals N separate single-adapter forwards (max abs error ≤ 1e-6 in float32); adapter index −1 equals the base model; adapters load from diffusion-omni LoRA checkpoints; a sampling run with mixed adapters matches per-adapter sampling with the same seeds | kernels (punica / S-LoRA style), hot-swap, a server; upstream to SGLang |
| L3 | edge | `torch.export` of the 1–4 step sampler for an image and a text bundle; exported samples match `load_bundle` samples within 1e-5 on CPU for fixed noise; the program reloads in a fresh process | MLX, Core ML, ExecuTorch, quantisation (optional dependencies and devices; no promise of on-device numbers) |
| L4 | workflow runtime | a graph result equals the same calls made directly through `Pipeline` and `REWARDS`; cycles, unknown node types and dangling inputs are rejected with clear errors; `--plan` prints the execution order without running | ComfyUI JSON import, a serving plan for SGLang |
| L5 | safety and provenance | watermark detection rate ≥ 0.99 on watermarked toy images with false-positive rate ≤ 0.01 on clean ones, at PSNR ≥ 40 dB; the provenance manifest hash changes when any input changes and verifies otherwise | concept erasure, robustness to compression and crops, C2PA |
| L6 | causal video | changing frames after k leaves the model's output for frames up to k bit-identical (causality test); training with per-frame t runs on `moving_blob`; rolling generation of 2x the trained frame count runs, and `direction_acc` on the generated frames is reported | Self-Forcing / causal distillation, KV-cache streaming, real video |

Every new package follows the repo checklist: scope doc in `docs/scope/<layer>.md`, then a package with `diffusion_omni.plugins` and `diffusion_omni.commands` entry points, a toy config, correctness tests and a README row. Core stays torch + numpy + pyyaml; any extra dependency is optional and lives in the new package.

**Cut rule.** If a layer has not passed review by the end of Tue 27, it ships as its scope doc plus a status of "not started" in GOAL_AND_PLAN, not as half-built code. L6 is the most likely cut, since it touches the core attention path.

## Workstreams

Owner names refer to the roles in [EXECUTION.md](docs/plan/EXECUTION.md): **impl-S** = Sonnet low effort; **impl-C** = Codex `gpt-5.6-luna`; **mech-H** = Haiku low effort; **review-C** = Codex review, read-only, low effort; **orch** = the orchestrating session; **user** = the repo owner.

| id | workstream | owner | review | done when |
|---|---|---|---|---|
| R1f | fix R1: override mode accepts registered reward names only (no `module:object`, no `plugins`); request and sample-count limits enforced; deterministic batching test (no wall-clock asserts) | impl-S | review-C, then escalated (security) | a test shows that an override naming `os:system` or any import path returns 400; all R-design validation rules have tests; the test passes 50 runs in a loop |
| R2f | fix R2: classify by array rank and the declared modality, not by frame count; cast bfloat16 to float32 before `.npy`; keep `""` as `""` | impl-S | review-C | a test per finding |
| R3f | finish R3: commit the worktree, move the `datasets.py` change into R3's own files or justify it, run the study | impl-S | review-C | R-design tests (perfect agreement, ties, inverse ranking, shrinking CI); script under 15 CPU-min; `docs/notes/R3.md` with the result |
| R4f | fix R4: reject NaN (write `null` and count it), deduplicate or reject repeated seeds, validate `-n > 0` and `--nfe` | impl-S | review-C | a test per finding; schema test still passes |
| E1–E7 | experiments in [BENCHMARK_RUNS.md](docs/plan/BENCHMARK_RUNS.md) | orch | review-C on the results doc | tables in `docs/results/`, raw JSON next to them |
| S0 | scope docs for L1–L6 (one page each: interface, files, tests, deferred items) | impl-C | review-C, user skims | six files in `docs/scope/` |
| L1–L6 | the v0 layers above | L1, L4, L5: impl-S; L2, L3, L6: impl-C | review-C; L6 escalated (core change) | exit criteria above |
| P1 | Phase D cost estimate (model, dataset, GPU type, hours, price, what number is reproduced) | orch | user signs off | `docs/plan/PHASE_D_COST.md` |
| G1 | `g2_kl_sweep.py`: `--teacher` and `--student` flags, so E4 can use the ts10 teacher | mech-H | review-C | flags default to today's paths; existing output unchanged |
| DOC | README rewrite against the code; GOAL_AND_PLAN status and layer map; scope §8 rows | mech-H drafts, orch edits | review-C | every command and path in README exists (checked by a script) |
| RN | rename to `diffusion-omni` (`diffusion_omni.*`, CLI `diffusion-omni` / `domni`), once the user signs off [RENAME.md](docs/plan/RENAME.md) | impl-C | review-C | the verification script in RENAME.md passes |
| V | final verification: fresh clone, `uv sync`, full tests, `tiny_config` smoke of every config, `--help` of every CLI command | mech-H | orch | a clean log in `docs/plan/VERIFY.md` |

## Merge order

1. R4f, R1f, R3f, R2f (as planned before: R2 codes against R1's client).
2. G1, S0 (docs and a script flag only).
3. L1, L5, L2, L3, L4 (independent packages; L4 last because it calls reward and pipeline code).
4. L6 (touches core; merged last among the layers so a revert is cheap).
5. DOC, then results docs from the experiments.
6. RN, only after everything else, so no open branch has to be rebased across the rename.
7. V on the final `main`.

Each merge: rebase the branch onto `main`, `uv run pytest -q` green on the merged tree, then follow the merge policy in EXECUTION.md.

## Day by day

| date | build | runs (CPU, orch) | gate |
|---|---|---|---|
| Sun 18 | plan docs (this file, `docs/plan/*`); user decisions below; start R1f, R2f, R3f, R4f | E1: 3-seed default report (60 student runs), started in the background | user answers the open decisions |
| Mon 19 | R fixes reviewed by Codex and patched; S0 scope drafts; G1 | E1 continues; E0 sanity score of the video ts10 teacher | R1f escalated review done |
| Tue 20 | merge R4f, R1f, R3f, R2f; S0 reviewed | E2: ts10 3-seed report (36 runs); E5: R3 study; E6: R1 throughput | 165+ tests green after each merge |
| Wed 21 | start L1–L6 in parallel worktrees | E3: D2 advantage mode, 3 seeds | |
| Thu 22 | L1–L6 coding | E4: order study on the ts10 teacher | orch rewrites `docs/results/distillation_toy.md` from E1 and E2 |
| Fri 23 | L1, L4, L5 into review | rerun anything that failed | scope §8 closed |
| Sat 24 | L2, L3 into review; patches | | |
| Sun 25 | merge L1, L5, L2, L3 | | |
| Mon 26 | L4 merge; L6 into escalated review | E7: layer toy checks (watermark ROC, causal toy run) | |
| Tue 27 | L6 patches and merge, or cut; P1 cost estimate | | **cut rule applies at end of day** |
| Wed 28 | DOC: README, GOAL_AND_PLAN, ARCHITECTURE package map | | user signs off P1 or defers it |
| Thu 29 | RN (rename to `diffusion-omni`) if signed off, else skipped | | rename verification passes |
| Fri 30 | V: fresh-clone verification; draft issue #2 and #3 updates for the user to post | | |
| Sat 31 | buffer; final merge per policy | | `main` green, every layer done or deferred in writing |

Slack: Fri 23–Sat 24 and Sat 31 absorb one slipped workstream. If two slip, L6 is cut first, then L4.

## Decisions needed from the user

1. Confirm the standing approval for merging reviewed workstream branches into `main` of this repo (EXECUTION.md, "Commit and push policy"). Without it, every merge waits for an explicit yes.
2. Sign off the `diffusion-omni` rename plan in [RENAME.md](docs/plan/RENAME.md): the `domni` CLI alias (`dfo` is taken on PyPI), and the one-release compatibility shims. The PyPI name `diffusion-omni` already belongs to an unrelated diff toolkit.
3. Accept the v0 scope and the cut rule for L1–L6, or move layers to deferred now.
4. After Tue 27: sign off or defer the Phase D cost estimate.

## Status

| id | status |
|---|---|
| D1–D7, C1, ARCH | merged (`9fe1202`) |
| R1, R2, R4 | coded; Codex findings open; not merged |
| R3 | partial, uncommitted in its worktree |
| video ts10 teacher | trained; not scored |
| E1–E7 | not run |
| S0, L1–L6, G1, P1, DOC, RN, V | not started |
