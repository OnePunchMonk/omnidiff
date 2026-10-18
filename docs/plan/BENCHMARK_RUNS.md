# Experiment runs still to do

Mirrors the checklist in [issue #3](https://github.com/OnePunchMonk/diffusion-omni/issues/3) and is the source for updating it; the user posts the updates. CPU only unless marked. Last updated 2026-10-18.

## How the estimates were made

"Process-hours" are wall-clock hours of one training process with default torch threads on this machine (15 cores). From `runs/logs/*.log`, single toy distill runs took 190–1120 s with in-run evaluation (mean about 450 s). The most recent run (consistency, `report_seeds.log`) trained at about 10 steps/s. Two processes in parallel roughly halve the wall time; more than three slow each other down and the agents' test runs. The estimates include scoring, which is a few minutes per teacher group.

## Order and estimates

| order | id | run | issue #3 item | runs | process-hours | wall (2 parallel) | depends on | result goes to |
|---|---|---|---|---|---|---|---|---|
| 1 | E1 | 3-seed report on the default teachers: `uv run python scripts/distill_toy_report.py --seeds 0 1 2`. Groups: image flow 7 students, ddpm 4, video 5, text 4 | "3-seed distill report on the current teachers"; also covers the D1, D3, D4 and coverage-config items with seeds | 20 x 3 = 60 | 5–8 | 3–4 h | nothing; start first | `docs/results/distillation_toy.md`, raw JSON beside it |
| 2 | E0 | sanity score of the video ts10 teacher (scorecard at 1/2/4/8/32 NFE, check `direction_acc`) | "Train the video teacher with `time_scale: 10`": **trained** (`runs/video_flow_ts10/sft/final`, 8000 steps, 481 s); only the check is left | 1 | 0.1 | | nothing | a line in the E2 section |
| 3 | E2 | 3-seed report on the ts10 teachers: `--teacher-suffix _ts10 --groups image_flow_grpo video_flow_grpo` | "3-seed distill report on the `time_scale: 10` teachers" | 12 x 3 = 36 | 3–5 | 2–3 h | E0 | same doc, "smooth teacher" section |
| 4 | E3 | D2 at 3 seeds: gradient mode at weights 0, 0.3, 1 (seeds 1 and 2 added to the existing seed 0) and advantage mode (`mode: advantage`, `group_size` 4) at weights 0.3, 1, 3, 300 steps each | "D2: reward-regularised DMD2"; adds the advantage-mode run D2 left open | 6 + 9 = 15 | 1.5 | 0.75 h | nothing | `docs/notes/D2.md`, then the results doc |
| 5 | E4 | order study on the ts10 teacher: `g2_kl_sweep.py` arms A and B, kl in {0, 0.03, 0.1, 0.3}, seeds 0 1 2, teacher `runs/image_flow_ts10/sft/final`, student `runs/distill/image_flow_dmd2_ts10` | "Order study repeated on the `time_scale: 10` teacher" | 2 x 4 x 3 = 24 RL runs | 1.5–2 | 1 h | G1 (the script hardcodes the teacher and student paths today) | `docs/results/g2_kl_sweep.md`, new section |
| 6 | E5 | R3 reliability: agreement (Spearman with bootstrap CI, every toy reward against the dataset oracles) and hacking probes (RL on reward A, tracking held-out B and the oracle) | "R3: reward agreement and hacking probes" | 1 script, 3–4 short RL runs | 0.25–0.5 | | R3f merged | `docs/notes/R3.md`, then a results doc |
| 7 | E6 | R1 throughput: requests/s and p50/p95 latency against direct calls, for batch limits 1, 8, 32 and 1, 4, 16 concurrent clients | "R1: reward service throughput" | 1 script | 0.2 | | R1f merged | `docs/notes/R1.md` |
| 8 | E7 | new-layer toy checks: watermark detection rate and false-positive rate on 1024 toy images (L5); causal `moving_blob` run, rolling 2x length, `direction_acc` (L6); multi-LoRA batched against sequential timing (L2) | not in #3 yet; add it | 3 short runs | 0.5–1 | | L2, L5, L6 | each layer's `docs/notes/L<n>.md` |

**CPU total: about 12–18 process-hours, about 8–10 wall hours at two in parallel, between Sun 18 and Mon 26.**

Before E1: the four `runs/distill/*_s*` dirs from the stopped run predate the D7 trainer change, so the harness will stop on them as stale. Either delete them or pass `--retrain-stale`.

Items in #3 that are already done as single-seed runs, written up in `docs/notes/`: D1 (`D1.md`: with `last_stage_prob 0.5`, the lowest data FD is at the trained 4 steps), D2 gradient mode (`D2.md`), D3 (`D3.md`), D4 (`D4.md`). E1 and E3 add their seeds.

## GPU (deferred)

| id | run | status |
|---|---|---|
| P1 / Phase D | one small real model per modality; reproduce one published distillation number | **no run.** A written cost estimate (`docs/plan/PHASE_D_COST.md`, due Tue 27) goes to the user for sign-off first. The compute budget is tight, so the default answer is deferred |

## Checklist for issue #3 (draft; the user posts it)

Distillation (toy, CPU)
- [ ] E1: 3-seed report, default teachers (60 runs, 5–8 process-hours)
- [x] video teacher with `time_scale: 10` trained (8000 steps); [ ] E0 sanity score
- [ ] E2: 3-seed report, ts10 teachers (36 runs, 3–5 process-hours)
- [x] D1: single seed, see `docs/notes/D1.md`; seeds via E1
- [ ] E3: D2 at 3 seeds, gradient and advantage modes
- [x] D3: single seed, see `docs/notes/D3.md`; seeds via E1
- [x] D4: single seed, negative, see `docs/notes/D4.md`; seeds via E1
- [ ] coverage configs scored against tuned teachers: via E1
- [ ] E4: order study on the ts10 teacher (needs G1)

Reward harness (toy, CPU)
- [ ] E5: R3 agreement and hacking probes
- [ ] E6: R1 throughput

New layers (toy, CPU)
- [ ] E7: watermark ROC, causal rolling generation, multi-LoRA batching

Real scale (GPU, needs a written cost estimate and sign-off first)
- [ ] Phase D cost estimate written; [ ] signed off; [ ] run
