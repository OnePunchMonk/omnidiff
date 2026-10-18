# Reference: diffusion-omni-data

Package: `packages/data`, import name `diffusion_omni.data`. Depends on diffusion-omni-core.

## Purpose

Data sources for SFT and distillation. The built-in sets are synthetic, so each one also knows the exact generating process and can score samples against it.

## Public API

Module `diffusion_omni.data.datasets`:

| name | role |
|---|---|
| `Blobs(sample_shape, num_classes=4)` | registered as `blobs` |
| `MovingBlob(sample_shape, num_classes=4)` | registered as `moving_blob` (video) |
| `TextLines(codec, lines=None, path=None)` | registered as `text_lines`; the built-in corpus when `lines` and `path` are unset |

Each class implements `sample(n, rng) -> (x0, cond)` and `oracle(samples, prompts) -> dict` (class or direction accuracy, shape error, coverage).

## CLI

None. Datasets are used by `sft.dataset` and `distill.data` in a config, and by `diffusion-omni scorecard` when the reference config names a dataset.

## Config keys

```yaml
sft:
  dataset: {type: blobs, num_classes: 4}          # or {type: moving_blob, ...} or {type: text_lines, path: data.txt}
distill:
  data: teacher                                    # or any DATASETS spec
```

`text_lines` with `path:` reads one line per example.

## Registries

`DATASETS`: `blobs`, `moving_blob`, `text_lines`.

## Extension points

Any object with `sample(n, rng) -> (x0, cond)` works. Point a config at it with `type: my_pkg.data:MyData`, or register it in `DATASETS`. `oracle` is optional; without it the scorecard reports only the generic metrics.

Tests: exercised through `packages/posttrain/tests/test_posttrain_e2e.py` and `packages/reward/tests/test_distill_report.py`. There are no dedicated data tests.
