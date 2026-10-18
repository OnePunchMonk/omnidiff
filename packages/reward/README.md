# diffusion-omni-reward

Reward functions registered in `REWARDS` (`channel_dominance`, `brightness`, `temporal_smoothness`, `char_fraction`, `contains`, `composite`), set-level metrics (Frechet distance, precision and recall, within-prompt diversity), aggregation over seeds, and the distillation scorecard.

```bash
uv run diffusion-omni scorecard --ref runs/<teacher>/sft/final --ckpt name=runs/<student>/distill/final --nfe 1 2 4 --out runs/scorecard
```

Reference: [docs/reference/reward.md](../../docs/reference/reward.md). Guide: [docs/guides/eval-scorecard.md](../../docs/guides/eval-scorecard.md).
