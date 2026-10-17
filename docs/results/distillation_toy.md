# Few-step distillation on the toy tasks: interim notes (2026-10-17)

CPU-only, tiny DiTs on synthetic data. These are single-seed, preliminary results; the full tables come from
`scripts/distill_toy_report.py` once all runs finish.

## Text (masked diffusion, 32-step teacher, 4-step students, n = 384)

Validity at 4 steps: teacher 0.14, teacher at temperature 0.6 (tuned baseline) 0.21, **KD 0.24**, **SDTT 0.08**.

- KD's gain over the tuned teacher is small, about what temperature sharpening buys (consistent with arXiv 2609.33947).
- SDTT gets worse with every round. Few-step failure here comes from tokens decoded together being sampled independently, and per-position targets cannot fix that. This motivated `coupled_kd` (noise-conditioned), whose run is in progress.

Full table:

| model | nfe | diversity | valid | coverage | reward | ms_per_sample |
|---|---|---|---|---|---|---|
| reference@32 | 32 | 0.3157 | 0.7135 | 1 | 0.2817 |  |
| reference | 1 | 0.9926 | 0 | 0 | 0.2768 | 0.5456 |
| reference | 2 | 0.9847 | 0.01562 | 0.2222 | 0.2744 | 1.129 |
| reference | 4 | 0.8674 | 0.1406 | 0.7778 | 0.2775 | 2.142 |
| reference | 8 | 0.6296 | 0.3672 | 1 | 0.276 | 4.297 |
| reference[T0.8] | 1 | 0.9923 | 0 | 0 | 0.2807 | 0.5323 |
| reference[T0.8] | 2 | 0.9588 | 0.02083 | 0.2222 | 0.2772 | 1.03 |
| reference[T0.8] | 4 | 0.8051 | 0.1719 | 0.7778 | 0.2769 | 2.164 |
| reference[T0.8] | 8 | 0.5748 | 0.4167 | 1 | 0.2793 | 4.329 |
| reference[T0.6] | 1 | 0.969 | 0 | 0 | 0.2893 | 0.5456 |
| reference[T0.6] | 2 | 0.9312 | 0.03385 | 0.2778 | 0.2822 | 1.098 |
| reference[T0.6] | 4 | 0.7412 | 0.2135 | 0.7222 | 0.2778 | 2.095 |
| reference[T0.6] | 8 | 0.5355 | 0.4505 | 1 | 0.2807 | 4.257 |
| kd | 1 | 0.9952 | 0.002604 | 0.05556 | 0.273 | 0.548 |
| kd | 2 | 0.934 | 0.04948 | 0.2222 | 0.2754 | 1.046 |
| kd | 4 | 0.7233 | 0.2396 | 0.7778 | 0.2721 | 2.132 |
| kd | 8 | 0.5247 | 0.4714 | 0.8889 | 0.2728 | 4.208 |
| sdtt | 1 | 1 | 0 | 0 | 0.267 | 0.5399 |
| sdtt | 2 | 0.9922 | 0.007812 | 0.05556 | 0.267 | 1.087 |
| sdtt | 4 | 0.9172 | 0.07812 | 0.5556 | 0.2693 | 2.173 |
| sdtt | 8 | 0.7325 | 0.2552 | 0.9444 | 0.273 | 4.239 |

## Image (flow teacher, 4-step students; in-training evals at the end of training, n = 128)

| method | fd vs teacher | precision | recall | class_acc | shape_err |
|---|---|---|---|---|---|
| kd | 0.087 | 1.00 | 1.00 | 0.92 | 0.048 |
| consistency | 0.120 | 0.92 | 0.96 | 0.95 | 0.056 |
| flow_map | 0.121 | 1.00 | 0.98 | 0.88 | 0.048 |
| dmd2 | 0.423 | 0.82 | 0.95 | 0.97 | 0.038 |
| dmd2 + GAN | 0.451 | 0.82 | 0.98 | 0.97 | 0.041 |

DMD2 is the most class-faithful and makes the sharpest blobs, but it drifts away from the teacher during training
(fd 0.23 -> 0.42): mode-seeking. Data-relative metrics are pending. Gaussian DMD2 without x0 clipping diverged
(class_acc 0.66 -> 0.35); a clipped rerun is in progress. The video teacher was retrained (8000 SFT steps)
because the old one ignored its direction label, and the video distills are being redone.
