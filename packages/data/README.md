# diffusion-omni-data

Data sources for SFT and distillation, registered in `DATASETS`. A dataset is any object with `sample(n, rng) -> (x0, cond)`. Built-in sets are synthetic: `blobs`, `moving_blob` (video) and `text_lines`. Each one also has an exact `oracle` that scores generated samples against the generating process.

Point a config at your own class with `type: my_pkg.data:MyData`.

Reference: [docs/reference/data.md](../../docs/reference/data.md).
