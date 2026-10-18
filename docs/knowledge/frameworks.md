# Frameworks that interoperate

Date read: 2026-10-18, via `gh api` (repo metadata, README, listed docs). Stars and dates are as of that read.

| repo | licence | stars | created | last push | note |
|---|---|---|---|---|---|
| [NVlabs/FastGen](https://github.com/NVlabs/FastGen) | Apache-2.0 | 1031 | 2026-01-15 | 2026-08-21 | distillation |
| [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo) | Apache-2.0 | 4613 | 2024-10-24 | 2026-10-18 | video post-train + inference |
| [radixark/miles_diffusion](https://github.com/radixark/miles_diffusion) | Apache-2.0 | 113 | 2026-05-11 | 2026-10-18 | RL post-training |
| [NVlabs/rcm](https://github.com/NVlabs/rcm) | Apache-2.0 | 823 | 2025-10-09 | 2026-06-25 | rCM, Causal-rCM |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) `python/sglang/multimodal_gen` | Apache-2.0 | 37262 (whole repo) | 2024-01-08 | 2026-10-18 | serving |
| [huggingface/diffusers](https://github.com/huggingface/diffusers) | Apache-2.0 | 34747 | 2022-05-30 | 2026-10-16 | library; latest release v0.41.0 (2026-10-06) |

## FastGen
README: PyTorch framework for ">=10B" training across T2I, I2V, V2V. Method table lists CM, sCM, TCM, MeanFlow; DMD2, f-Distill, LADD, CausVid, Self-Forcing; KD and CausalKD. Networks: EDM, EDM2, DiT, SD 1.5, SDXL, Flux, Qwen-Image; WAN (T2V, I2V, VACE), CogVideoX, Cosmos Predict2. DDP and FSDP2. `fastgen/methods/` contains `consistency_model`, `distribution_matching`, `fine_tuning`, `knowledge_distillation`; no RL directory. Evaluation is via scripts (FID mentioned in `scripts/README.md`, not read).

## FastVideo
README: unified post-training and real-time inference for video; DMD2 stepwise distillation, Video Sparse Attention, sparse distillation (">50x denoising speedup" as the README states it), Self-Forcing causal distillation, FSDP2. News (2026-08/09/10) lists FastH3 checkpoints (4-step and 8-step DMD2-based) distilled from MiniMax-H3, made with Nuva Lab and the NVIDIA FastGen team. No RL or discrete support in the README.

## miles_diffusion
README (News 2026-08): Flow-GRPO, DiffusionNFT and SFT under one trainer; rollouts served by SGLang-diffusion; training on FSDP2; recipes for SD3.5, Qwen-Image, Wan2.2-T2V-A14B, LTX-2.3, Cosmos3, MiniMax H3. [docs/fsdp_model_roles.md](https://github.com/radixark/miles_diffusion/blob/main/docs/fsdp_model_roles.md): "one trainable actor", optional frozen reference and teacher; separate instances are "not cross-family distillation or a new DMD2 loss"; the teacher "has no built-in consumer" so `--teacher-load` needs custom prepare/loss hooks.

## rCM
See distillation-methods.md. README states native PyTorch and a FlashAttention-2 JVP kernel with FSDP/CP support.

## SGLang Diffusion
README: inference framework for image/video; broad model list (Wan, FLUX, Qwen-Image, LTX, MiniMax-H3, FastH3, ...); OpenAI-compatible API, CLI, ComfyUI plugin; NVIDIA, AMD, Intel XPU, Ascend, Apple MPS. It states its major components "are based on a fork of FastVideo on Sept. 24, 2025". Roadmap [issue 43313](https://github.com/sgl-project/sglang/issues/43313) (open, created 2026-10-09, updated 2026-10-12): a "lossy `high` tier" combining sparse attention, few-step distillation "(PDD, Turbo LoRA, on top of FastH3 V2)" and FP8/NVFP4 for at least 2x over `exact`; keep distilled LoRAs effective when merged; "extend the quality gate" to Blackwell GPUs and LoRA weights. The issue body I read does not name SSIM or PSNR.

## diffusers
README: pipelines, interchangeable schedulers, model building blocks, training guides. The README does not mention distillation or RL; I did not check its training examples folder.

## Relevance to this repo

- Reference-behaviour comparison targets (re-implemented, not depended on): `packages/distill/src/diffusion_omni/distill/methods/`.
- Export with sampler spec toward diffusers/SGLang: `packages/distill/src/diffusion_omni/distill/export.py`.
- RL trainer with a single trainable role plus reference, comparable to miles's actor model: `packages/posttrain/src/diffusion_omni/posttrain/rl/trainer.py`; the distil trainer adds multiple roles: `packages/distill/src/diffusion_omni/distill/trainer.py`.
