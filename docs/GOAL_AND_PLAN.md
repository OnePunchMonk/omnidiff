# omnidiff: goal and plan

Last updated 2026-10-17.

## Goal

**omnidiff is a diffusion monorepo for text, image and video.** It covers the layers of the diffusion stack that the established frameworks leave open, on one small, exact, PyTorch + NumPy core.

The two reference points:

- **SGLang Diffusion owns serving**: kernels, attention backends, caching, quantisation, parallelism, disaggregated stages, LoRA hot-swap.
- **miles-diffusion owns large-scale RL/SFT post-training**: FSDP2, Flow-GRPO, DiffusionNFT, rollouts served by SGLang.

omnidiff does not compete with either. It builds the layers between and around them, and produces artifacts they consume: students that SGLang serves, and rewards and scorecards that miles calls.

What makes a monorepo worth it over a set of scripts is that every layer shares the same core:

- one `DiffusionProcess` interface (masked, Gaussian and flow), where every reverse step is a policy with an exact log-probability;
- one modality-agnostic model contract (adapter → backbone → adapter);
- registries and entry points, so installing a package makes its components buildable from config;
- toy-scale configs and correctness tests that run the whole thing on a laptop CPU.

## Principles

1. **Exactness over features.** Every new method lands with a test of the property that makes it correct: on-policy ratio = 1, a closed-form log-prob, marginal preservation, a JVP against finite differences. The test comes before any benchmark claim.
2. **`omnidiff-core` stays torch + numpy (+ pyyaml).** Other packages may add optional dependencies, such as VLM judges in `reward` or MLX in an edge package, but never into core.
3. **Plug and play by default.** Components go in registries, are built from `{type: ..., **kwargs}`, and accept shared context injection. A package joins by declaring the `omnidiff.plugins` and `omnidiff.commands` entry points.
4. **Toy first, scale on evidence.** Each layer proves itself on CPU toys, then reproduces one published number at small real scale. No GPU run happens without a written cost estimate.
5. **Interop over reinvention.** Where a framework already owns a layer (FastGen for large-scale continuous distillation, SGLang for serving), target its formats and contribute upstream.

## Layer map

What SGLang Diffusion and miles-diffusion miss, re-checked 2026-10-17. Two corrections to the earlier analysis: NVIDIA FastGen now covers continuous few-step distillation at scale, and miles_diffusion ships some built-in reward scorers.

| layer | package | status | who else is there | why / why not here |
|---|---|---|---|---|
| Core: processes, models, sampling, inference | `omnidiff-core` | **done (v0.1)** | n/a | the shared foundation |
| RL/SFT post-training | `omnidiff-posttrain` | **done (v0.1)**, toy-verified for all three modalities | miles-diffusion (scale) | the research bed: same algorithms at toy scale, plus text diffusion |
| **Few-step distillation** | `omnidiff-distill` | **v0 in progress**: KD, DMD2, consistency, flow map, SDTT, scorecard ([scope](scope/distillation.md)) | FastGen, FastVideo, rcm (continuous, at scale) | open for text, for RL ∘ distillation, and for student evaluation |
| Reward and eval harness | `omnidiff-reward` | seed (toy rewards) | GenEval, VBench and others, scattered; miles has some scorers | unified, batched, reliability-aware; the backend miles could call |
| Data engine | `omnidiff-data` | seed (toy datasets) | NeMo curator, img2dataset | lightweight recaption, filter, dedupe, latent precompute |
| Causal / streaming video training | later (`omnidiff-causal`) | not started | FastVideo, rcm (Self-Forcing, Causal-rCM) | where world models are heading; grows out of distill |
| Multi-tenant LoRA (batching many adapters in one pass) | later | not started | SGLang (load, merge, unmerge only) | less solved; overlaps serving, so upstream to SGLang |
| Edge deployment (MLX, Core ML, ExecuTorch) | later | not started | mflux, DiffusionKit, FastVideo MLX | fragmented; few-step students are what makes edge viable |
| Workflow runtime (ComfyUI graph → serving plan) | later | not started | ComfyUI wrappers | medium gap |
| Safety and provenance (erasure, watermarking) | later | not started | research code | medium gap |
| Pretraining framework, VAE/tokenizer training | **out of scope** | n/a | VeOmni, torchtitan work | lab-scale compute; doesn't fit the budget |

## Plan

| milestone | contents | exit criteria | compute |
|---|---|---|---|
| **M0: foundation** ✅ | core (masked, Gaussian, flow; DiT; text/image/video adapters; CFG; LoRA), posttrain (SFT, GRPO, DDPO, KL), uv workspace, entry-point plugins, CLI | 31 CPU tests green; reward rises for all four shipped configs | CPU |
| **M1: distillation v0** (in progress) | Phases A and B of the scope doc: few-step samplers in core, multi-role `DistillTrainer`, KD, DMD2, SDTT, consistency (CD→sCM), flow map, scorecard v0 | exit criteria in scope doc §5 | CPU |
| **M2: RL ∘ distillation** | Phase C: GRPO on few-step students, reward-regularised DMD2, order-effect study | a written result, positive or negative | CPU, maybe 1 small GPU |
| **M3: reward/eval harness** | batched reward service, a reward-reliability study (agreement and hacking probes), scorecard promoted to a shared eval; adapter for miles' reward flags | miles-diffusion can call an omnidiff reward via its plug-in flags | low |
| **M4: real-scale anchor** | Phase D: one small real model per modality (for example an MDLM-class dLLM, ImageNet-64 EDM2); reproduce one published distillation number | within noise of the paper | **GPU: estimate and ask first** |
| **M5: handoff** | export students and sampler specs to diffusers/SGLang format; upstream issue or PR | a student served by SGLang Diffusion | low |

After M5, choose the next layer from the map: causal video (it follows from distillation) or the data engine.

## Repo conventions

- `packages/<layer>/` holds a `pyproject.toml`, `src/omnidiff/<layer>/` (PEP 420 namespace) and `tests/`.
- `uv sync` installs everything editable; `uv run pytest` runs every package's tests (a few seconds on CPU).
- `configs/*.yaml` hold one config per recipe; `docs/scope/*.md` hold one scope doc per layer before any code lands.
- New layer checklist: scope doc → package with entry points → toy config → correctness tests → README row.

## Open decisions

Distillation decisions are recorded in [scope/distillation.md §7](scope/distillation.md#7-decisions-2026-10-17). Still open: the Phase D target model and where its compute comes from.
