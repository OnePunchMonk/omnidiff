# RL for diffusion, and reward hacking

Date read: 2026-10-18. All entries are **abstract-only reads** of the arXiv abs page.

## Algorithms

- **DDPO**, Black et al., 2023-05-22. [2305.13301](https://arxiv.org/abs/2305.13301). Denoising as a multi-step MDP with policy gradients; beats reward-weighted likelihood; tasks: compressibility, aesthetics, VLM-feedback prompt alignment. No numbers in abstract.
- **Flow-GRPO**, Liu et al., v1 2025-05-08, v5 2025-10-27. [2505.05470](https://arxiv.org/abs/2505.05470). ODE-to-SDE conversion preserves marginals so exploration is possible; Denoising Reduction cuts training steps while keeping inference steps. SD3.5-M GenEval 63% to 95%; visual text rendering 59% to 92%. Authors report "very little" reward hacking.
- **DanceGRPO**, Xue et al., 2025-05-12. [2505.07818](https://arxiv.org/abs/2505.07818). GRPO across diffusion and rectified flows, 3 tasks, 4 foundation models, 5 reward models; "up to 181%" over baselines on HPS-v2.1, CLIP Score, VideoAlign, GenEval.
- **DiffusionNFT**, Zheng et al., v1 2025-09-19, v2 2026-02-16, ICLR 2026 Oral. [2509.16117](https://arxiv.org/abs/2509.16117). Online RL on the forward process via flow matching; contrasts positive and negative samples; no likelihood estimation, CFG-free, works with black-box solvers. Claims up to 25x more efficient than Flow-GRPO; GenEval 0.24 to 0.98 within 1k steps versus Flow-GRPO 0.95 with over 5k steps and CFG.

## Reward hacking

- Jena et al., *Elucidating Optimal Reward-Diversity Tradeoffs*, 2024-09-09. [2409.06493](https://arxiv.org/abs/2409.06493). Argues reward hacking is unavoidable; finds limits of KL and LoRA-scaling regularisers; proposes inference-time Annealed Importance Guidance.
- **GARDO**, He et al., 2025-12-30. [2512.24138](https://arxiv.org/abs/2512.24138). Proxy reward rises while quality drops and diversity collapses; regularises only a high-uncertainty subset, refreshes the reference model periodically, boosts rewards for high-quality diverse samples.
- Hong et al., *Understanding Reward Hacking in Text-to-Image RL*, v1 2026-01-06 (v2 2026-08-31). [2601.03468](https://arxiv.org/abs/2601.03468). Aesthetic/preference and consistency rewards each get hacked; ensembling only partially helps; common failure is artifact-prone images; proposes an artifact reward model as regulariser.

Contrast: Flow-GRPO reports little hacking, the later papers report it as a main failure. These are different reward models and setups; I did not reconcile them.

## Relevance to this repo

- GRPO/DDPO-style algorithms with exact log-probs: `packages/posttrain/src/diffusion_omni/posttrain/rl/algorithms.py`, `rl/trainer.py`.
- Few-step RL study, KL anchor against collapse: `scripts/g2_kl_sweep.py`, [docs/results/g2_kl_sweep.md](../results/g2_kl_sweep.md). GARDO's periodically-refreshed reference is an alternative to a fixed KL anchor; not implemented here.
- Diversity/coverage tracking for hacking: `packages/reward/src/diffusion_omni/reward/scorecard.py`.
- DiffusionNFT (forward-process, no log-probs) has no counterpart in `rl/algorithms.py` as far as the file list shows; not checked line by line.
