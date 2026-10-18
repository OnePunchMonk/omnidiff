# Guide: sft

Supervised denoising training. This produces the base policy that RL starts from, and the teacher that distillation starts from.

## Tiny run (CPU)

```bash
uv run diffusion-omni sft -c configs/image_flow_grpo.yaml device=cpu sft.steps=200 sft.batch_size=16 output_dir=runs/guide/img
uv run diffusion-omni sft -c configs/text_masked_grpo.yaml device=cpu sft.steps=200 sft.batch_size=16 output_dir=runs/guide/txt
```

Each prints `saved runs/guide/<name>/sft/final`. The loss is logged to `runs/guide/<name>/sft/log.jsonl`.

## Full run

```bash
uv run diffusion-omni sft -c configs/image_flow_grpo.yaml
```

Without overrides the config runs 2000 steps at batch 64. The output goes to `runs/image_flow_grpo/sft/final`.

## Keys you will change

| key | what it does |
|---|---|
| `sft.steps`, `sft.batch_size`, `sft.lr` | length and size of the run |
| `sft.dataset` | `{type: blobs, num_classes: 4}`, `moving_blob` (video), `text_lines` (text) or your own class |
| `sft.cond_dropout` | fraction of examples trained unconditionally (needed for CFG) |
| `sft.save_every` | save `step_<n>` checkpoints every n steps (0 saves only `final`) |
| `sft.init_from` | start from an existing checkpoint |

Any key can be set on the command line, for example `sft.lr=1e-4`.

Reference: [posttrain](../reference/posttrain.md).
