# Few-step distillation on the toy tasks

CPU-only, tiny DiTs (dim 128, depth 4) on synthetic data whose generating process is known exactly. Single seed.
These results check how the methods behave relative to each other and to tuned baselines. They are not claims
about real-scale models. Reproduce with `scripts/distill_toy_report.py` (tables below) and
`scripts/g2_order_study.sh` (order study).

## Summary

1. **Text: noise-conditioned distillation (`coupled_kd`) is the only method that fixes few-step masked decoding.**
   - Validity at 4 steps: 0.66 for `coupled_kd`, against 0.20 for the tuned teacher (T = 0.6), 0.27 for KD, 0.10 for SDTT and 0.72 for the teacher at 32 steps.
   - At 1 step `coupled_kd` reaches 0.45, where the teacher scores 0.00. At 8 steps it reaches 0.74, matching the 32-step teacher.
   - Coverage stays at 0.94. Diversity is 0.35 at 4 steps against 0.29 for the 32-step teacher, so it isn't collapsing.
   - Per-position methods barely move or hurt: KD is roughly temperature sharpening, and SDTT degrades every round. That fits the diagnosis that the few-step failure is independent sampling of tokens decoded together, not bad marginals.
2. **Image and video: at 4 or more steps, the teacher with a tuned sampler beats every student.**
   - Rectified flow with time shift 3 reaches teacher-relative FD 0.069 for images and 0.057 for video at 4 steps.
   - Students win only at 1–2 steps. For images, KD at 1 step reaches FD 0.043, consistency at 1 step 0.073 and DMD2 at 2 steps 0.089; the best teacher sampler is at 1.03 and 0.19. For video, DMD2 at 1 step reaches 0.42 against 1.11.
3. **Sample quality versus coverage, at 4 steps:**
   - DMD2 and rCM give the most class-faithful, sharpest samples (class_acc 0.98–0.99) but lose recall (0.73–0.87). This is the mode-seeking signature.
   - The renoise sampler alone shows the same trade on the undistilled teacher: precision 0.98, recall 0.72.
   - KD keeps recall near 0.97.
4. **Negative results:**
   - **The flow-map (MeanFlow identity) student behaves like the teacher's Euler sampler at every step count.** It did not learn long jumps in 2000 steps.
   - **DMD2 on the Gaussian (VP, epsilon) teacher fails** (class_acc 0.23–0.36), even with x0 clipping. The DDPM teacher is itself weak here (class_acc 0.52 at 32 steps).
   - **DMD2 is best at 2 steps, not at the 4-step grid it was trained on**, and degrades badly at 8 steps, which it never saw.

## Order study: RL and distillation (image flow teacher, redness reward)

Same RL budget for every arm (40 epochs, LoRA rank 8, no KL). Scored at 4 steps against the SFT teacher.

| arm | reward | precision | recall | class_acc | diversity |
|---|---|---|---|---|---|
| DMD2 student, no RL | 0.031 | 0.84 | 0.87 | 0.98 | 0.178 |
| A: RL on the teacher (at its own 10 steps) | 0.091 | 0.80 | 0.96 | 0.80 | 0.159 |
| B: distil, then RL on the 4-step student | **0.287** | 0.32 | 0.93 | 0.58 | 0.214 |
| C: RL, then distil (DMD2) | 0.117 | 0.76 | **0.59** | 0.75 | 0.158 |

- **B** optimises the reward about 3x faster than RL on the teacher (the 4-step renoise student is an exact, high-variance policy) and reward-hacks much harder.
- **C** keeps precision but loses coverage: recall 0.59, against 0.87 for DMD2 on the SFT teacher. That fits the hypothesis that distillation's mode-seeking compounds RL's collapse.
- **Neither order dominates.** Next arm: B with a KL penalty to the distilled student (the reference is free, since it is the student with LoRA disabled).

## Open issues

- **Flow-map objective:** check the adaptive weighting, the t/s sampling and the training length on long jumps. Compare the JVP and finite-difference variants at scale.
- **DMD2's best step count differs from its trained grid:** investigate backward simulation and the grid.
- **Scale up `coupled_kd`.** On real tokenizers the per-token Gumbel input is V-dimensional, so it needs a low-rank or top-k encoding of the noise.
- **Seeds and confidence intervals** before any of this is stated more strongly.

## Full tables

Columns are defined in `scripts/distill_toy_report.py`. In short: `fd`, `precision` and `recall` are against the teacher at 32 steps; `data_*` are against real data; `class_acc`, `direction_acc`, `shape_err`, `motion_err`, `center_tv`, `valid` and `coverage` are exact oracles on the synthetic data; `reference[...]` rows are tuned teacher samplers.

### Teacher `image_flow_grpo`

| model | nfe | diversity | fd | precision | recall | data_fd | data_precision | data_recall | class_acc | shape_err | center_tv | reward | ms_per_sample |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference@32 | 32 | 0.1644 | -1.455e-08 | 1 | 1 | 0.08892 | 0.9336 | 0.9258 | 0.9355 | 0.0326 | 0.04102 | 0.02728 |  |
| reference | 1 | 0.1741 | 1.03 | 0.001953 | 0.8047 | 1.22 | 0.001953 | 0.7344 | 0.8145 | 0.1471 | 0.1914 | 0.05341 | 0.2071 |
| reference | 2 | 0.1288 | 0.3673 | 0.8496 | 0.7988 | 0.593 | 0.834 | 0.709 | 0.8789 | 0.04845 | 0.08594 | 0.04808 | 0.4061 |
| reference | 4 | 0.1512 | 0.1542 | 0.9395 | 0.9336 | 0.3272 | 0.9277 | 0.9004 | 0.9023 | 0.04499 | 0.07617 | 0.03969 | 0.7991 |
| reference | 8 | 0.1593 | 0.0729 | 0.9453 | 0.9434 | 0.2059 | 0.9316 | 0.9492 | 0.9102 | 0.04221 | 0.06641 | 0.03411 | 1.624 |
| reference[renoise] | 1 | 0.1741 | 1.03 | 0.001953 | 0.8047 | 1.22 | 0.001953 | 0.7344 | 0.8145 | 0.1471 | 0.1914 | 0.05341 | 0.3169 |
| reference[renoise] | 2 | 0.09641 | 0.4954 | 0.9141 | 0.5293 | 0.7213 | 0.8574 | 0.4434 | 0.9336 | 0.04468 | 0.1016 | 0.03987 | 0.5621 |
| reference[renoise] | 4 | 0.1149 | 0.2722 | 0.9805 | 0.7148 | 0.4165 | 0.9531 | 0.6719 | 0.9844 | 0.0272 | 0.03711 | 0.0254 | 0.9375 |
| reference[renoise] | 8 | 0.1277 | 0.1886 | 0.9883 | 0.748 | 0.2924 | 0.9844 | 0.7051 | 0.9883 | 0.0166 | 0.0332 | 0.01906 | 1.658 |
| reference[shift3] | 1 | 0.1741 | 1.03 | 0.001953 | 0.8047 | 1.22 | 0.001953 | 0.7344 | 0.8145 | 0.1471 | 0.1914 | 0.05341 | 0.2155 |
| reference[shift3] | 2 | 0.1232 | 0.1895 | 0.7363 | 0.7539 | 0.3401 | 0.75 | 0.6387 | 0.8945 | 0.06462 | 0.07812 | 0.04135 | 0.4136 |
| reference[shift3] | 4 | 0.1471 | 0.06943 | 0.9082 | 0.9219 | 0.1873 | 0.9219 | 0.916 | 0.9258 | 0.04444 | 0.07031 | 0.03148 | 0.8342 |
| reference[shift3] | 8 | 0.1588 | 0.04129 | 0.9277 | 0.9414 | 0.1414 | 0.9395 | 0.9512 | 0.9199 | 0.04112 | 0.06836 | 0.02825 | 1.822 |
| image_flow_kd | 1 | 0.1582 | 0.04305 | 0.8906 | 0.9727 | 0.1453 | 0.8984 | 0.9668 | 0.9492 | 0.04421 | 0.06641 | 0.02947 | 0.2142 |
| image_flow_kd | 2 | 0.1738 | 0.07525 | 0.8906 | 0.9766 | 0.1806 | 0.8887 | 0.9805 | 0.9375 | 0.04219 | 0.06836 | 0.03029 | 0.4225 |
| image_flow_kd | 4 | 0.1826 | 0.1288 | 0.8652 | 0.9727 | 0.2349 | 0.8652 | 0.9805 | 0.9258 | 0.04752 | 0.06836 | 0.0315 | 0.9919 |
| image_flow_kd | 8 | 0.1794 | 0.1197 | 0.8477 | 0.9707 | 0.2024 | 0.8516 | 0.9609 | 0.9141 | 0.07111 | 0.07031 | 0.02714 | 1.639 |
| image_flow_consistency | 1 | 0.1654 | 0.07327 | 0.6152 | 0.9883 | 0.169 | 0.6504 | 0.9883 | 0.9395 | 0.06369 | 0.07617 | 0.03027 | 0.2106 |
| image_flow_consistency | 2 | 0.1616 | 0.06792 | 0.7754 | 0.9512 | 0.1655 | 0.8125 | 0.9434 | 0.9453 | 0.05963 | 0.07617 | 0.02789 | 0.4181 |
| image_flow_consistency | 4 | 0.1543 | 0.08325 | 0.8789 | 0.9043 | 0.1784 | 0.8496 | 0.9062 | 0.9863 | 0.05143 | 0.06836 | 0.02318 | 0.8416 |
| image_flow_consistency | 8 | 0.1527 | 0.1046 | 0.9062 | 0.8984 | 0.195 | 0.9062 | 0.8496 | 0.9883 | 0.04343 | 0.01953 | 0.02119 | 1.744 |
| image_flow_flow_map | 1 | 0.1781 | 1.085 | 0.001953 | 0.8418 | 1.285 | 0 | 0.7598 | 0.8125 | 0.1439 | 0.1836 | 0.05201 | 0.2353 |
| image_flow_flow_map | 2 | 0.1358 | 0.3791 | 0.8359 | 0.8477 | 0.6137 | 0.8164 | 0.7402 | 0.8926 | 0.04784 | 0.08984 | 0.04746 | 0.4468 |
| image_flow_flow_map | 4 | 0.1596 | 0.196 | 0.9277 | 0.9375 | 0.385 | 0.8984 | 0.918 | 0.9082 | 0.04525 | 0.08203 | 0.04213 | 0.8096 |
| image_flow_flow_map | 8 | 0.164 | 0.08021 | 0.9238 | 0.957 | 0.2142 | 0.9141 | 0.9551 | 0.916 | 0.04201 | 0.06836 | 0.03459 | 1.632 |
| image_flow_dmd2 | 1 | 0.1159 | 0.2705 | 0.7676 | 0.7188 | 0.343 | 0.7227 | 0.7285 | 0.9512 | 0.06275 | 0.1055 | 0.01788 | 0.2068 |
| image_flow_dmd2 | 2 | 0.1508 | 0.08878 | 0.8809 | 0.9043 | 0.1976 | 0.8926 | 0.9355 | 0.9727 | 0.03548 | 0.07422 | 0.02429 | 0.4475 |
| image_flow_dmd2 | 4 | 0.1781 | 0.3633 | 0.8438 | 0.873 | 0.4761 | 0.834 | 0.8066 | 0.9805 | 0.03963 | 0.1426 | 0.03125 | 0.823 |
| image_flow_dmd2 | 8 | 0.2006 | 1.485 | 0.4922 | 0.6465 | 1.575 | 0.3945 | 0.5918 | 0.9688 | 0.05963 | 0.3145 | 0.04051 | 1.668 |
| image_flow_dmd2_gan | 1 | 0.1153 | 0.2678 | 0.7441 | 0.7305 | 0.3371 | 0.7168 | 0.7109 | 0.9512 | 0.06566 | 0.1133 | 0.01541 | 0.2082 |
| image_flow_dmd2_gan | 2 | 0.1521 | 0.09046 | 0.8535 | 0.916 | 0.2001 | 0.8887 | 0.9414 | 0.9707 | 0.03299 | 0.08398 | 0.02293 | 0.4207 |
| image_flow_dmd2_gan | 4 | 0.1773 | 0.4064 | 0.8457 | 0.8477 | 0.5259 | 0.8418 | 0.7969 | 0.9863 | 0.03427 | 0.1348 | 0.0323 | 0.9465 |
| image_flow_dmd2_gan | 8 | 0.1991 | 1.575 | 0.4766 | 0.5293 | 1.672 | 0.3691 | 0.457 | 0.9766 | 0.06057 | 0.3066 | 0.04203 | 1.894 |
| image_flow_rcm | 1 | 0.1605 | 0.827 | 0.001953 | 0.7422 | 0.9996 | 0.001953 | 0.6309 | 0.8574 | 0.1405 | 0.1719 | 0.04371 | 0.2116 |
| image_flow_rcm | 2 | 0.09633 | 0.4336 | 0.9121 | 0.5645 | 0.6367 | 0.875 | 0.457 | 0.9531 | 0.05648 | 0.1035 | 0.03375 | 0.4184 |
| image_flow_rcm | 4 | 0.1142 | 0.2492 | 0.9883 | 0.7285 | 0.3688 | 0.957 | 0.6816 | 0.9863 | 0.03813 | 0.05273 | 0.02015 | 0.8411 |
| image_flow_rcm | 8 | 0.1266 | 0.1971 | 0.9922 | 0.7422 | 0.2772 | 0.9961 | 0.6992 | 0.9883 | 0.02672 | 0.05273 | 0.01414 | 1.734 |

### Teacher `image_ddpm_ddpo`

| model | nfe | diversity | fd | precision | recall | data_fd | data_precision | data_recall | class_acc | shape_err | center_tv | reward | ms_per_sample |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference@32 | 32 | 0.2089 | -4.717e-09 | 1 | 1 | 0.9658 | 0.373 | 0.9004 | 0.5234 | 0.1736 | 0.0332 | 0.09984 |  |
| reference | 1 | 0.5632 | 22.72 | 0 | 1 | 25.7 | 0 | 0.9629 | 0.6738 | 0.402 | 0.4473 | 0.3771 | 0.2031 |
| reference | 2 | 0.2478 | 4.605 | 0.2461 | 0.791 | 7.262 | 0 | 0.5469 | 0.7383 | 0.1996 | 0.1602 | 0.2177 | 0.3999 |
| reference | 4 | 0.2044 | 0.3308 | 0.6758 | 0.8027 | 1.576 | 0.1172 | 0.6973 | 0.666 | 0.1498 | 0.05078 | 0.1168 | 0.7931 |
| reference | 8 | 0.2037 | 0.1147 | 0.7852 | 0.8242 | 1.101 | 0.252 | 0.8145 | 0.6055 | 0.157 | 0.04102 | 0.1049 | 1.586 |
| reference[renoise] | 1 | 0.5632 | 22.72 | 0 | 1 | 25.7 | 0 | 0.9629 | 0.6738 | 0.402 | 0.4473 | 0.3771 | 0.2023 |
| reference[renoise] | 2 | 0.3802 | 11.51 | 0 | 0.9883 | 14.74 | 0 | 0.9434 | 0.875 | 0.2848 | 0.2773 | 0.3041 | 0.4049 |
| reference[renoise] | 4 | 0.1248 | 0.3328 | 0.9512 | 0.4863 | 0.9033 | 0.1875 | 0.2383 | 0.6992 | 0.1465 | 0.04297 | 0.09305 | 0.8117 |
| reference[renoise] | 8 | 0.1057 | 1.433 | 0.9902 | 0.3945 | 0.9167 | 0.5879 | 0.3711 | 0.5957 | 0.1954 | 0.04688 | 0.04638 | 1.63 |
| image_ddpm_dmd2 | 1 | 0.005129 | 4.958 | 0.9492 | 0.08008 | 3.204 | 0.005859 | 0 | 0.2324 | 0.00498 | 0.9883 | 0.0001068 | 0.2035 |
| image_ddpm_dmd2 | 2 | 0.1763 | 0.6674 | 0.4199 | 0.6523 | 1.044 | 0.05664 | 0.502 | 0.3223 | 0.1652 | 0.08398 | 0.09339 | 0.4058 |
| image_ddpm_dmd2 | 4 | 0.1943 | 1.458 | 0.6172 | 0.291 | 2.632 | 0.09375 | 0.1309 | 0.3457 | 0.1883 | 0.1523 | 0.1488 | 0.8104 |
| image_ddpm_dmd2 | 8 | 0.1937 | 3.899 | 0.7148 | 0.1426 | 5.391 | 0.08594 | 0.04688 | 0.3613 | 0.1863 | 0.2461 | 0.1723 | 1.773 |

### Teacher `video_flow_grpo`

| model | nfe | diversity | fd | precision | recall | data_fd | data_precision | data_recall | direction_acc | shape_err | motion_err | reward | ms_per_sample |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference@32 | 32 | 0.1558 | -9.622e-09 | 1 | 1 | 0.1026 | 0.9102 | 0.8867 | 0.998 | 0.02476 | 0.1003 | 0.01335 |  |
| reference | 1 | 0.1293 | 1.108 | 0.005859 | 0.125 | 1.071 | 0.003906 | 0.0918 | 1 | 0.1497 | 0.1421 | 0.01261 | 0.2252 |
| reference | 2 | 0.1114 | 0.2521 | 0.748 | 0.6035 | 0.4446 | 0.8047 | 0.4512 | 1 | 0.02962 | 0.1249 | 0.01556 | 0.4316 |
| reference | 4 | 0.1405 | 0.08574 | 0.8613 | 0.8887 | 0.2324 | 0.8945 | 0.8359 | 1 | 0.02297 | 0.1024 | 0.01377 | 0.8302 |
| reference | 8 | 0.1502 | 0.05147 | 0.8828 | 0.8945 | 0.1736 | 0.9238 | 0.8848 | 1 | 0.02135 | 0.102 | 0.01387 | 1.671 |
| reference[renoise] | 1 | 0.1293 | 1.108 | 0.005859 | 0.125 | 1.071 | 0.003906 | 0.0918 | 1 | 0.1497 | 0.1421 | 0.01261 | 0.2197 |
| reference[renoise] | 2 | 0.09004 | 0.4177 | 0.7754 | 0.248 | 0.6489 | 0.8008 | 0.1426 | 1 | 0.04496 | 0.11 | 0.01667 | 0.4518 |
| reference[renoise] | 4 | 0.1192 | 0.2062 | 0.9336 | 0.5859 | 0.414 | 0.9297 | 0.5215 | 1 | 0.02919 | 0.07493 | 0.008417 | 0.9029 |
| reference[renoise] | 8 | 0.1252 | 0.1695 | 0.9629 | 0.6035 | 0.3625 | 0.9473 | 0.5156 | 1 | 0.02242 | 0.06738 | 0.0076 | 1.789 |
| reference[shift3] | 1 | 0.1293 | 1.108 | 0.005859 | 0.125 | 1.071 | 0.003906 | 0.0918 | 1 | 0.1497 | 0.1421 | 0.01261 | 0.2225 |
| reference[shift3] | 2 | 0.1245 | 0.136 | 0.7695 | 0.7285 | 0.2549 | 0.8164 | 0.6719 | 1 | 0.04723 | 0.1249 | 0.01845 | 0.4297 |
| reference[shift3] | 4 | 0.1416 | 0.0568 | 0.8691 | 0.8652 | 0.1709 | 0.9043 | 0.832 | 1 | 0.03136 | 0.1084 | 0.01351 | 0.8409 |
| reference[shift3] | 8 | 0.1516 | 0.03788 | 0.8926 | 0.8945 | 0.1333 | 0.9375 | 0.8594 | 0.998 | 0.02816 | 0.1011 | 0.01338 | 1.647 |
| video_flow_dmd2 | 1 | 0.1012 | 0.4225 | 0.7461 | 0.4824 | 0.5937 | 0.6426 | 0.3086 | 1 | 0.1008 | 0.1391 | 0.006451 | 0.2219 |
| video_flow_dmd2 | 2 | 0.1137 | 0.3735 | 0.8789 | 0.6406 | 0.5528 | 0.7363 | 0.5195 | 0.9961 | 0.1001 | 0.1199 | 0.00563 | 0.4554 |
| video_flow_dmd2 | 4 | 0.1274 | 0.3406 | 0.9531 | 0.6992 | 0.5163 | 0.7031 | 0.625 | 1 | 0.08657 | 0.1129 | 0.008934 | 0.8859 |
| video_flow_dmd2 | 8 | 0.1331 | 0.5902 | 0.8457 | 0.5059 | 0.701 | 0.4961 | 0.459 | 0.998 | 0.09147 | 0.1075 | 0.01373 | 1.849 |
| video_flow_flow_map | 1 | 0.1295 | 1.148 | 0.005859 | 0.1191 | 1.111 | 0.005859 | 0.08594 | 0.998 | 0.1466 | 0.1429 | 0.0146 | 0.2198 |
| video_flow_flow_map | 2 | 0.117 | 0.2207 | 0.7285 | 0.6934 | 0.4053 | 0.7812 | 0.5508 | 1 | 0.03416 | 0.1263 | 0.01825 | 0.4236 |
| video_flow_flow_map | 4 | 0.1451 | 0.07699 | 0.8613 | 0.8984 | 0.2113 | 0.8906 | 0.8633 | 1 | 0.02993 | 0.1 | 0.01467 | 0.8269 |
| video_flow_flow_map | 8 | 0.1535 | 0.05634 | 0.8809 | 0.9121 | 0.1786 | 0.9219 | 0.8906 | 1 | 0.01899 | 0.104 | 0.01494 | 1.637 |

### Teacher `text_masked_grpo`

| model | nfe | diversity | valid | coverage | reward | ms_per_sample |
|---|---|---|---|---|---|---|
| reference@32 | 32 | 0.2917 | 0.7188 | 1 | 0.2768 |  |
| reference | 1 | 0.9981 | 0 | 0 | 0.2861 | 0.2336 |
| reference | 2 | 0.9832 | 0.009766 | 0.1667 | 0.2819 | 0.4716 |
| reference | 4 | 0.842 | 0.1543 | 0.7222 | 0.2763 | 0.7441 |
| reference | 8 | 0.6013 | 0.3887 | 1 | 0.2794 | 1.36 |
| reference[T0.8] | 1 | 0.996 | 0 | 0 | 0.2909 | 0.1682 |
| reference[T0.8] | 2 | 0.9543 | 0.02148 | 0.2778 | 0.2813 | 0.3388 |
| reference[T0.8] | 4 | 0.781 | 0.1895 | 0.7222 | 0.2783 | 0.657 |
| reference[T0.8] | 8 | 0.5258 | 0.4414 | 1 | 0.281 | 1.305 |
| reference[T0.6] | 1 | 0.9712 | 0.003906 | 0.05556 | 0.2973 | 0.1735 |
| reference[T0.6] | 2 | 0.9227 | 0.0332 | 0.2778 | 0.2876 | 0.3292 |
| reference[T0.6] | 4 | 0.7341 | 0.2031 | 0.7222 | 0.282 | 0.6572 |
| reference[T0.6] | 8 | 0.4725 | 0.4727 | 0.9444 | 0.2808 | 1.401 |
| text_masked_kd | 1 | 0.991 | 0.005859 | 0.05556 | 0.2799 | 0.1861 |
| text_masked_kd | 2 | 0.9181 | 0.0625 | 0.4444 | 0.2822 | 0.3501 |
| text_masked_kd | 4 | 0.7 | 0.2695 | 0.6111 | 0.2805 | 0.697 |
| text_masked_kd | 8 | 0.4827 | 0.4883 | 0.8889 | 0.2815 | 1.346 |
| text_masked_sdtt | 1 | 1 | 0 | 0 | 0.271 | 0.1662 |
| text_masked_sdtt | 2 | 0.9982 | 0.007812 | 0.1667 | 0.2714 | 0.3432 |
| text_masked_sdtt | 4 | 0.9029 | 0.1016 | 0.7222 | 0.2687 | 0.748 |
| text_masked_sdtt | 8 | 0.7131 | 0.2871 | 1 | 0.2733 | 1.391 |
| text_masked_coupled_kd | 1 | 0.5655 | 0.4531 | 0.7222 | 0.2772 | 0.1303 |
| text_masked_coupled_kd | 2 | 0.4656 | 0.5488 | 0.8889 | 0.2754 | 0.2432 |
| text_masked_coupled_kd | 4 | 0.3498 | 0.6602 | 0.9444 | 0.2781 | 0.4716 |
| text_masked_coupled_kd | 8 | 0.2642 | 0.7402 | 0.9444 | 0.2775 | 0.8958 |
