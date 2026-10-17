# Build plan to Friday 2026-10-23

Tracking: [goals and plan, #2](https://github.com/OnePunchMonk/omnidiff/issues/2) and [benchmark runs, #3](https://github.com/OnePunchMonk/omnidiff/issues/3).

Two layers ship by Friday: **distillation complete** (milestones M1 and M2 in [docs/GOAL_AND_PLAN.md](docs/GOAL_AND_PLAN.md), plus the CPU half of Phase E), then **the reward and eval harness** (M3). Phase D, the real-scale reproduction, is out of this window: it needs GPU time and a cost estimate first.

## Definition of done

**Distillation** is done when every exit criterion for Phases A–C in [scope §5](docs/scope/distillation.md#5-phases-and-exit-criteria) is met and written up. That means:
- every method in scope §4 is implemented with a correctness test;
- the open issues in [docs/results/distillation_toy.md](docs/results/distillation_toy.md#open-issues) are closed or recorded as negative results;
- the headline tables are reported over 3 seeds;
- a distilled student exports to a documented, framework-neutral format.

**Reward** is done when:
- rewards can be served in batches over a stable API;
- a miles-diffusion adapter exists, written against miles' actual reward interface;
- a reliability study (agreement and hacking probes) has a written toy result;
- the scorecard runs as a general `omnidiff eval`.

## How the work runs

| role | model | what it does |
|---|---|---|
| implementer | Sonnet, low effort | one workstream per agent, in its own git worktree and branch (`ws/<id>`); writes code, tests and a notes file; commits and never pushes |
| reviewer | Opus, medium effort | reviews each branch's diff against this plan and the repo principles; writes findings ranked by severity |
| second reviewer, designer, implementer | Codex (`gpt-5.6-luna`) | an independent review of each branch; design critiques and interface specs; some coding workstreams (C1, ARCH, D7) |
| implementer (again) | the same Sonnet agent | patches every finding, or replies with a reason for not doing so |
| integrator | Haiku, low effort | merges reviewed branches into `main` in the order below, runs the full test suite, pushes; stops and reports on any conflict or failure |
| orchestrator | this session | writes the plan, starts the agents, resolves conflicts, runs long CPU experiments, writes `docs/results` |

Rules for every agent:
- Commit as `git -c user.name=OnePunchMonk -c user.email=aggarwal.avaya@yahoo.com commit ...`, with no co-author trailers and no session links.
- CPU only; no GPU runs.
- `uv run pytest -q` must pass before every commit. A new method or component ships with a test of the property that makes it correct (GOAL_AND_PLAN principle 1).
- Stay inside the files your workstream owns. Shared files (`methods/__init__.py`, `pyproject.toml` entry points) get one-line additions only.
- Do not edit `README.md`, `docs/results/` or `docs/scope/`. Put findings in `docs/notes/<id>.md`; the orchestrator folds them into the docs.
- Match the existing code style: same comment density, docstrings that explain *why*, registries and context injection, toy configs in `configs/`.
- Teachers for experiments live in the main checkout: `/Users/aggar/omnidiff/runs/...`. Keep any experiment an agent runs under 15 CPU-minutes.

## Wave 1: distillation (Sat 17 to Tue 20)

| id | workstream | owns | done when |
|---|---|---|---|
| D1 | **DMD2 grid mismatch.** DMD2 is best at 2 steps, not at its 4-step training grid, and collapses at 8. Investigate backward simulation (which grid times the student's own samples are noised to) and the student's time-sampling distribution; fix or explain. | `methods/dmd2.py`, `configs/distill/*dmd2*.yaml`, new tests | a test pins the backward-simulation states to the student grid; a short CPU run shows best quality at the trained NFE, or the notes explain why not |
| D2 | **Reward-regularised DMD2** (scope §4, the G2 thread). Add an optional reward term to DMD2's generator loss: a reward gradient for differentiable rewards, advantage-weighted otherwise. Add a differentiable (torch) path to the toy rewards that need one. | new `methods/dmd2_reward.py` (or an option in `dmd2.py`, coordinated with D1), `reward/rewards.py` torch path, config, tests | a test checks the reward gradient against autograd on a known reward; a toy config trains; notes give reward against precision/recall for 2–3 weights |
| D3 | **Progressive distillation** for continuous processes (scope §8, not started). Rounds of "1 student step = 2 teacher steps", halving the steps each round; Salimans and Ho target in x0 space. | new `methods/progressive.py`, config, tests | a test checks the target equals two exact teacher DDIM/Euler steps; rounds advance and the grid halves; a toy config trains |
| D4 | **Discrete consistency for text** (scope §4 `dcd`). Read arXiv 2607.19686 and 2609.33947 and implement a multi-step / multi-mask consistency objective for masked diffusion, compared against `coupled_kd`. | new `methods/dcd.py`, config, tests, `docs/notes/D4.md` with the paper reading | a correctness test of the objective's fixed point; a toy config trains; notes compare it with `coupled_kd` at 1/2/4/8 steps |
| D5 | **Export (Phase E, CPU half).** `omnidiff export --ckpt ... --out ...` writes a framework-neutral student bundle: weights (`model.safetensors` if `safetensors` is installed, `torch.save` otherwise), `sampler.json` (process, sampler, timesteps, guidance, provenance) and the model config. Add a loader and a round-trip test. | new `distill/export.py`, CLI subcommand, tests, `docs/notes/D5.md` with the schema | the round trip reproduces samples bit-for-bit on CPU; the schema is documented |
| D7 | **Exact save and resume** for `DistillTrainer`. Checkpoint every role's weights, optimiser and EMA, the RNG states (torch, numpy, generator), the iteration, method state (SDTT round, consistency target net, fake score) and the teacher-pool identity (seed and size, or the pool itself). | `trainer.py` (`save_state` / `resume_from`), method `state_dict` hooks, tests | a test shows that N uninterrupted steps and N/2 + resume + N/2 give bit-identical student weights for kd, dmd2 and sdtt |
| C1 | **Config coverage.** Distill configs for every method on every process it supports: video flow KD, consistency, rCM; DDPM-image KD and consistency. | new `configs/distill/*.yaml` | the end-to-end config test covers them |
| ARCH | **Architecture doc**, `docs/ARCHITECTURE.md`: package map, core abstractions, data flows, extension recipes, the invariants the tests enforce. | `docs/ARCHITECTURE.md` | it is accurate against the code, as checked by review |
| D6 | **Multi-seed harness.** `scripts/distill_toy_report.py --seeds 0 1 2` trains distill runs per seed (teachers fixed), aggregates mean ± std, and writes the same tables. Add `model.time_scale: 10` teacher variants of the image and video flow configs as new configs. Code only: the orchestrator does the long runs. | `scripts/distill_toy_report.py`, new configs | a dry-run test on tiny configs covers the aggregation |

Order: D1–D6, C1 and ARCH start in parallel. D2 and D1 both touch DMD2: D2 adds a subclass or a hook and leaves D1's code alone. D7 touches the trainer, so it rebases onto main after D1 and D3. The merge order is D5, D6, C1, ARCH, D3, D1, D2, D4, D7.

Evaluation provenance (from the design review), which D6 and R4 should follow: freeze the prompt IDs, the teacher checkpoint, the pool seed, the evaluation seeds, the NFE grid and the metric version, and write raw per-seed JSON before aggregating it into markdown.

After the merges (Tue 20 to Wed 21, orchestrator): 3-seed runs of all distill configs on both teacher time scales, then rewrite `docs/results/distillation_toy.md` and close scope §8.

## Wave 2: reward and eval harness (Wed 21 to Thu 22)

| id | workstream | owns | done when |
|---|---|---|---|
| R1 | **Batched reward service.** `omnidiff serve-reward --spec ...`: a stdlib HTTP server (no new hard dependencies) that batches requests up to a size or time limit and calls any `REWARDS` entry. Samples go in as base64 npy or text; JSON API plus a Python client. | new `reward/service.py`, `reward/client.py`, CLI, tests | a test starts the server on a free port and checks that batched results equal direct calls, including concurrent clients |
| R2 | **miles-diffusion adapter.** Read miles_diffusion's reward plug-in interface from its source (github.com/radixark/miles_diffusion) and write an adapter: an importable function or class with the exact signature miles expects, backed by an omnidiff reward or the R1 service. | new `reward/miles.py`, tests with a stub that mirrors the interface, `docs/notes/R2.md` citing the miles files and lines | the stub test passes; the notes quote the interface and the flags miles uses to load it |
| R3 | **Reward-reliability study.** (a) Agreement: rank correlation between rewards and the dataset oracles over shared samples. (b) Hacking probes: RL against reward A, tracking reward A, a held-out reward B and the oracle, to find where they diverge. Library code plus a script. | new `reward/reliability.py`, `scripts/reward_reliability.py`, tests | the metric code is tested on synthetic data with known correlation; the script runs on CPU in under 15 minutes; notes report the result |
| R4 | **`omnidiff eval`.** Promote the scorecard into eval suites: an `EVALS` registry; `omnidiff eval --suite ... --ckpt ...` with multi-seed mean ± CI; a stable JSON schema versioned in the output. | `reward/scorecard.py`, new `reward/eval.py`, `core/registry.py` (add `EVALS`), CLI, tests | the existing scorecard tests still pass; a schema test; CI widths shrink with n on a synthetic check |

Interface specs, file ownership and JSON schemas for R1–R4: [docs/notes/R-design.md](docs/notes/R-design.md). The miles hook is `async def custom_rm(args, samples, **kwargs) -> list[float]`, loaded with `--custom-rm-path`; R2 must verify it against a pinned miles revision.

Order: R1–R4 in parallel. R2 depends on R1's client API, so R2 codes against the client interface in R-design.md and the review checks the two match. The merge order is R4, R1, R3, R2.

## Friday 23

- Orchestrator: final runs, results docs, GOAL_AND_PLAN status, README.
- Integrator: final merge and push. The full test suite must pass on `main`.

## Status

| id | status |
|---|---|
| D5, D6 | coded; in review |
| C1, ARCH | coded by Codex; in review |
| D1–D4 | coding |
| D7 | coding (Codex) |
| R1–R4 | not started |
