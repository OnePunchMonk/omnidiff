# Few-step distillation methods

Date read: 2026-10-18. Unless marked otherwise, every entry comes from the **arXiv abstract page only** (abstract-only read; full text not read). Numbers are as the abstract states them. Existing coverage of the landscape is in [docs/scope/distillation.md](../scope/distillation.md); this file adds verified details and does not repeat that doc's framework table.

## Distribution matching

- **DMD** (One-step Diffusion with Distribution Matching Distillation), Yin et al., v1 2023-11-30, CVPR 2024. [2311.18828](https://arxiv.org/abs/2311.18828). Minimises an approximate KL whose gradient is the difference of two score functions (target diffusion model, and one fitted to the one-step generator's outputs), plus a regression loss on multi-step outputs. Reported: FID 2.62 ImageNet-64, 11.49 zero-shot COCO-30k, about 20 FPS (FP16).
- **DMD2** (Improved DMD), Yin et al., 2024-05-23. [2405.14867](https://arxiv.org/abs/2405.14867). Drops the regression loss and dataset construction; adds a two time-scale update rule for stability, a GAN loss against real images, and a training change for multi-step sampling (simulating inference-time generator inputs). Reported: FID 1.28 ImageNet-64 one-step, 8.35 zero-shot COCO 2014, SDXL distilled to megapixel output.

## Consistency models

- **CM**, Song et al., 2023-03-02, ICML 2023. [2303.01469](https://arxiv.org/abs/2303.01469). One-step FID 3.55 CIFAR-10, 6.20 ImageNet-64 (distillation); also trainable standalone.
- **iCT** (Improved Techniques), Song and Dhariwal, 2023-10-22. [2310.14189](https://arxiv.org/abs/2310.14189). Standalone consistency training, removes EMA from the teacher, pseudo-Huber loss instead of LPIPS, lognormal noise schedule, discretisation doubling. One-step FID 2.51 CIFAR-10, 3.25 ImageNet-64; two-step 2.24 and 2.77.
- **TCM** (Truncated CM), Lee et al., 2024-10-18, ICLR 2025. [2410.14895](https://arxiv.org/abs/2410.14895). Consistency training over a truncated time range with a two-stage procedure. Abstract gives no FID values; claims better 1- and 2-step FID than iCT-deep with networks more than 2x smaller.
- **sCM** (continuous-time CMs), Lu and Song, 2024-10-14, ICLR 2025. [2410.11081](https://arxiv.org/abs/2410.11081). Unifies diffusion/CM parameterisations, stabilises continuous-time training, scales to 1.5B on ImageNet-512. Two-step FID 2.06 CIFAR-10, 1.48 ImageNet-64, 1.88 ImageNet-512.
- **rCM** (score-regularised continuous-time consistency), Zheng et al., v1 2025-10-09, v3 2026-05-06, ICLR 2026. [2510.08431](https://arxiv.org/abs/2510.08431). Attributes sCM's weak fine detail to error accumulation and the mode-covering forward divergence; adds score distillation as a long-skip regulariser. FlashAttention-2 JVP kernel compatible with parallelism, >10B-parameter models, Cosmos-Predict2 and Wan2.1 up to 14B. Claims it "generally matches" DMD2 quality metrics with less mode collapse and better diversity, no GAN tuning, 1-4 steps, 15x-50x speedup. Repo [NVlabs/rcm](https://github.com/NVlabs/rcm) (Apache-2.0, 823 stars, last push 2026-06-25, read 2026-10-18); its README lists Causal-rCM and says training/inference is native PyTorch without accelerate or diffusers.

## Flow maps / mean-velocity

- **Shortcut models**, Frans et al., 2024-10-16. [2410.12557](https://arxiv.org/abs/2410.12557). One network, one training phase, conditioned on noise level and step size. Claims better than consistency models and reflow across step budgets. No numbers in abstract.
- **MeanFlow**, Geng et al., 2025-05-19. [2505.13447](https://arxiv.org/abs/2505.13447). Average-velocity identity; no pretraining, distillation or curriculum. 1-NFE FID 3.43 on ImageNet-256 from scratch.
- **Align Your Flow**, Sabour, Fidler, Kreis, 2025-06-17. [2506.14603](https://arxiv.org/abs/2506.14603). States (analytically and empirically) that consistency models get worse as steps increase while flow maps do not; two continuous-time objectives, autoguidance, optional adversarial finetuning. No FID in abstract. Abstract does not mention FLUX or LoRA; docs/scope says it distils FLUX with LoRA, which I could not verify from the abstract.
- **Flow Map Distillation Without Data**, Tong et al., [2511.19428](https://arxiv.org/abs/2511.19428), CVPR 2026. **Read via a web-search summary only** (the CVPR PDF returned 403, the arXiv page was not fetched). Summary states data-free distillation from the prior only, one-step FID 1.45 (ImageNet-256) and 1.49 (ImageNet-512) from SiT-XL/2+REPA. Treat as unverified until the abstract page is read.
- **Mean Flow Distillation**, Zhao et al., 2026-06-09. [2606.11155](https://arxiv.org/abs/2606.11155). Argues MFD acts as a temporal low-pass filter on VSD noise; claims a "Mean Flow Matching Theorem". Evaluated on 4D occupancy forecasting and text-to-image; no numbers in abstract. (Its domain is not the video/ImageNet setting scope/distillation.md implies.)

## Trajectory / regression

- **Progressive distillation**, Salimans and Ho, 2022-02-01, ICLR 2022. [2202.00512](https://arxiv.org/abs/2202.00512). Repeatedly halves sampler steps; from up to 8192 steps down to 4 with little loss; FID 3.0 in 4 steps on CIFAR-10; cost no more than training the original.
- **PDD** (Parallel Decoding Distillation), Shaul, Liu, Vahdat, Berner, 2026-07-28. [2607.26004](https://arxiv.org/abs/2607.26004). Predicts multiple denoising steps per evaluation, learns mean velocity "without regressing its derivative" (no JVP). Claims SOTA at 4-8 NFE on LTX-2.3, Wan 14B T2V, Qwen-Image. No numbers in abstract.
- **TMD** (Transition Matching Distillation), Nie et al., 2026-01-14, CVPR 2026. [2601.09881](https://arxiv.org/abs/2601.09881). Backbone plus flow head; DMD with flow-head rollout per transition; Wan2.1 1.3B and 14B. No numbers in abstract.

## Not fetched (listed in the reading list, no source read here)

DMD variants f-Distill (2502.15681), LADD (2403.12015), 1.x-Distill (2604.04018), AdvDMD (2604.28126), CausVid (2412.07772), Self Forcing (2506.08009), VSA (2505.13389). Claims about these in docs/scope remain unverified by this pass.

## Relevance to this repo

- DMD2, GAN head, reward-regularised variant: `packages/distill/src/diffusion_omni/distill/methods/dmd2.py`, `dmd2_reward.py`.
- Consistency / sCM / rCM: `methods/consistency.py`, `methods/rcm.py`; flow maps: `methods/flow_map.py`; KD and progressive: `methods/kd.py`, `methods/progressive.py`.
- Multi-role trainer (DMD2 needs two trainable roles): `packages/distill/src/diffusion_omni/distill/trainer.py`.
- Process helpers (`to_x0`, `ode_step`): `packages/core/src/diffusion_omni/core/processes/`.
- Note PDD's JVP-free mean-velocity route is a candidate alternative to the `torch.func.jvp` path in `flow_map.py`; not implemented here.
