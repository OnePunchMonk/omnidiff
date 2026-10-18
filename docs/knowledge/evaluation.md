# Evaluation metrics

Date read: 2026-10-18. All entries are **abstract-only reads** of the arXiv abs page.

- **FID / TTUR**, Heusel et al., 2017-06-26, NeurIPS 2017. [1706.08500](https://arxiv.org/abs/1706.08500). Introduces FID, a Frechet distance between feature statistics of real and generated images, argued to capture similarity better than Inception Score. (The same paper proves TTUR convergence; DMD2 borrows the "two time-scale" name for a different update rule, see distillation-methods.md.)
- **Improved Precision and Recall**, Kynkaanniemi et al., 2019-04-15, NeurIPS 2019. [1904.06991](https://arxiv.org/abs/1904.06991). Non-parametric manifold estimates that report quality (precision) and coverage (recall) separately; also a per-sample perceptual quality estimate and a truncation analysis. The abstract does not state k-NN radii or k; implementation detail (k) is not verified.
- **Stein et al., metrics vs diffusion models**, 2023-06-07, NeurIPS 2023. [2306.04675](https://arxiv.org/abs/2306.04675). In a large human study no existing metric correlates strongly with human realism; FID's Inception-V3 extractor unfairly penalises diffusion models; DINOv2-ViT-L/14 is a better extractor; a library of 17 metrics over 9 encoders. Implication: an FD on a small random feature net is only a toy proxy.
- **GenEval**, Ghosh et al., 2023-10-17. [2310.11513](https://arxiv.org/abs/2310.11513). Object-focused compositional evaluation (co-occurrence, position, count, colour) using detectors.
- **VBench**, Huang et al., 2023-11-29. [2311.17982](https://arxiv.org/abs/2311.17982). 16 video-quality dimensions with human-preference alignment checks.
- **GroupEval** (2609.33947): joint quality and across-output diversity for text. See discrete-diffusion.md.

## Relevance to this repo

- Frechet distance, precision/recall (k=3 default), within-prompt diversity, `RandomFeatures` extractor: `packages/reward/src/diffusion_omni/reward/metrics.py`.
- Teacher-relative and data-relative scorecard: `packages/reward/src/diffusion_omni/reward/scorecard.py`.
- `RandomFeatures` is a random-feature net, not Inception or DINOv2, so FD values from the scorecard are not comparable with published FIDs (e.g. DMD2's 1.28).
- GenEval and VBench have no implementation here; the toy datasets use their own `oracle()`.
