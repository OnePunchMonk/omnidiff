# Rename proposal

Status: **done in the working tree on 2026-10-18** (GitHub repo renamed to `OnePunchMonk/diffusion-omni`; code renamed without compat shims since nothing has been released). Uncommitted until approved.

## Why rename

- **The PyPI name is taken.** `omnidiff` 1.3 on PyPI is "A diff toolkit for python" (github.com/game1024/omnidiff). The distribution names `omnidiff-core` and the rest are free today, but `pip install omnidiff` would install someone else's package, and the top-level import name `omnidiff` would collide with it.
- **"diff" reads as text diffing.** `omnidiff` and `unidiff` on PyPI are both diff tools.

## Primary: `diffusion-omni`

| item | old | new |
|---|---|---|
| GitHub repo | `OnePunchMonk/omnidiff` | `OnePunchMonk/diffusion-omni` |
| PyPI distributions | `omnidiff-core`, `-data`, `-reward`, `-posttrain`, `-distill`; workspace `omnidiff-workspace` | `diffusion-omni-core`, `diffusion-omni-data`, `diffusion-omni-reward`, `diffusion-omni-posttrain`, `diffusion-omni-distill` (and `-causal`, `-lora`, `-edge`, `-workflow`, `-safety` for the new layers); workspace `diffusion-omni-workspace` |
| import namespace (PEP 420; must be an identifier, so underscore) | `omnidiff.core`, `omnidiff.distill`, ... | `diffusion_omni.core`, `diffusion_omni.distill`, ... |
| CLI | `omnidiff` | `diffusion-omni`, with the short alias `domni` |
| entry-point groups | `omnidiff.plugins`, `omnidiff.commands` | `diffusion_omni.plugins`, `diffusion_omni.commands` |
| schema strings | `omnidiff.eval/v1`, `omnidiff.reward/v1` | `diffusion_omni.eval/v1`, `diffusion_omni.reward/v1` |
| extras in messages | `omnidiff-distill[export]` | `diffusion-omni-distill[export]` |

### Collision check (2026-10-18)

The check used the PyPI JSON API (`https://pypi.org/pypi/<name>/json`, where 404 means free), `gh api repos/OnePunchMonk/<name>` and GitHub name searches.

| name | PyPI | GitHub |
|---|---|---|
| `diffusion-omni`, `diffusion_omni` (PyPI normalises both to the same name), `diffusion-omni-core` | free | `OnePunchMonk/diffusion-omni` free; no repo named `diffusion-omni` |
| CLI alias `dfo` | **taken**: `dfo` 0.0, "PDFO - Powell's Derivative-Free Optimization solvers" | no repo of note by that name; not on PATH or in Homebrew here |
| CLI alias `domni` (proposed) | free | r4vanan/domni (2 stars, unrelated) |
| alternative alias `dfom` | free | three repos with 0–2 stars |

`dfo` is rejected: the PyPI name belongs to PDFO, and a console script called `dfo` could clash with it. `domni` is the proposed short alias.

### Overlap with well-known "omni" projects

The name is free, but it sits close to several active projects. Docs should say plainly what this repo is, to avoid confusion:

- **VITA-MLLM/Omni-Diffusion** (156 stars, ICML 2026): unified multimodal understanding and generation with masked discrete diffusion. This is the closest: same words reversed, and it also uses masked diffusion.
- **llm-conditioned-diffusion/OmniDiffusion** (14 stars) and smaller "omni-diffusion" repos.
- **vllm-project/vllm-omni** (7.2k stars, `vllm-omni` on PyPI): inference for omni-modality models, including diffusion. Same "serving framework + omni" space.
- **sgl-project/sglang-omni** (1.5k stars): SGLang serving for audio and unified multimodal models. SGLang is one of the two reference points in GOAL_AND_PLAN.
- **Qwen3-Omni** (4k stars) and **OmniGen2** (4.1k stars): "Omni" model names in multimodal generation.

None of them uses `diffusion-omni` or `diffusion_omni`. The README's first line should state the scope: a PyTorch + NumPy monorepo for the layers around diffusion training, not an omni-modal model.

## Alternatives (if the user changes their mind)

Checked the same way. All are free on PyPI as `<name>` and `<name>-core`, and free as `OnePunchMonk/<name>`.

| name | rationale | GitHub repos with the name |
|---|---|---|
| `noisepolicy` | every reverse step is a policy with an exact log-prob | none found |
| `diffmdp` | "every diffusion sampler is an MDP", the README's one idea | none found |
| `tridiff` | three processes and three modalities | two 0-star repos |
| `polydiff` | several modalities on one core | three repos with 1–2 stars |
| `denoisekit` | a kit of denoising-model layers | one 1-star repo |

Rejected: `diffstack` (NVlabs/diffstack, 114 stars), `unidiff`, `triflow`, `tessera`, `anneal`, `diffusionlab` and `noisebridge` (taken on PyPI).

## Mechanical checklist

Done on one branch `ws/RN` after every other branch is merged, in this order. Each step ends with `uv run pytest -q` green.

1. **Python namespace.** In every package, run `git mv packages/<pkg>/src/omnidiff packages/<pkg>/src/diffusion_omni`. These are PEP 420 namespace dirs: there is no `src/omnidiff/__init__.py` and none may be added. Then rewrite imports `omnidiff.` → `diffusion_omni.` in `packages/`, `scripts/`, `examples/` and `conftest.py`. That is 38 Python files in `packages/` today, plus the scripts and examples.
2. **Distribution names** in `packages/*/pyproject.toml` and the root `pyproject.toml`:
   - `omnidiff-<pkg>` → `diffusion-omni-<pkg>` everywhere it appears: `name`, `dependencies` lists, `[tool.uv.sources]`;
   - `omnidiff-workspace` → `diffusion-omni-workspace`;
   - `[tool.hatch.build.targets.wheel] packages = ["src/diffusion_omni"]`;
   - the `omnidiff-distill[export]` extra named in error messages (`distill/export.py`).
3. **Entry-point groups.** Every `[project.entry-points."omnidiff.plugins"]` becomes `"diffusion_omni.plugins"`, and `"omnidiff.commands"` becomes `"diffusion_omni.commands"`. Update `PLUGIN_GROUP` and `COMMAND_GROUP` in `core/registry.py` to match.
4. **CLI.** In `[project.scripts]`, set `diffusion-omni = "diffusion_omni.core.cli:main"` and `domni = "diffusion_omni.core.cli:main"`. Change `prog="omnidiff"` to `prog="diffusion-omni"` in `core/cli.py`.
5. **Config keys.** No YAML key contains `omnidiff` today; configs mention it only in comments (`# omnidiff sft -c ...`). Rewrite the comments to `diffusion-omni sft -c ...`. Any `type: omnidiff.x:Class` or `plugins: [omnidiff...]` added later must be rewritten too: `grep -rn "omnidiff" configs`.
6. **Schema strings.** Rename `omnidiff.eval/v1` (R4, `reward/eval.py`) and `omnidiff.reward/v1` (R1, `reward/service.py`) to `diffusion_omni.eval/v1` and `diffusion_omni.reward/v1`. The bundle's `sampler.json` uses an integer `schema_version` with no name, but the bundle README text says "omnidiff student bundle" and needs rewording.
7. **Checkpoints and bundles on disk.** Existing `runs/*/config.yaml` files contain no `omnidiff` strings (checked). A `plugins` entry in a bundle's `sampler.json` would; the loader maps it (step 11).
8. **Docs.** Update the README, `docs/GOAL_AND_PLAN.md`, `docs/ARCHITECTURE.md`, `docs/scope/*`, `docs/notes/*` and `docs/results/*` (commands only, not the numbers), plus `plan.md`, `docs/plan/*`, the package READMEs and the `__init__.py` docstrings.
9. **Lock file.** Regenerate `uv.lock` (40 `omnidiff` references today) with `uv lock`. Do not hand-edit it.
10. **GitHub.** These steps are done by the user, or with their approval:
    - rename the repo to `OnePunchMonk/diffusion-omni` in Settings (GitHub redirects the old URL);
    - run `git remote set-url origin` in the main checkout;
    - update the issue links in `plan.md`;
    - update the titles and bodies of #2 and #3 where they name the project.
11. **Backward-compatibility shims.** These ship in release 0.2.x only and are removed after it.
    - Import alias: a small `omnidiff-compat` distribution (`packages/compat/`) with `src/omnidiff/<sub>/__init__.py` files that re-export `diffusion_omni.<sub>` and emit a `DeprecationWarning`. A `sys.modules` alias would need a meta-path finder to cover submodules, so the explicit shim package is simpler to test.
    - Old CLI name: `omnidiff = "diffusion_omni.core.cli:main_deprecated"`, which prints one warning line and then calls `main`.
    - Old entry-point groups: `discover_plugins` and the CLI command loader read both the `diffusion_omni.*` and `omnidiff.*` groups. They warn once for each package found under an old group and deduplicate by entry-point value.
    - Schemas: readers treat `omnidiff.eval/v1` and `omnidiff.reward/v1` as equal to the new strings. Writers emit only the new ones.
    - Bundles: `load_bundle` maps `plugins` entries that start with `omnidiff.` to `diffusion_omni.`, with a warning.
12. **Verification.** Run the script below, then the full test suite, then the shims' own tests: import `omnidiff.core`, run `omnidiff --help`, load a plugin under the old group, and read an old-schema eval JSON.

## Verification script

Saved as `scripts/check_rename.py` in the RN branch:

```python
"""Fail if any reference to the old name survives outside the allowed places."""
import re, subprocess, sys

OLD = "omnidiff"
NEW_PKG, NEW_CLI, ALIAS = "diffusion_omni", "diffusion-omni", "domni"
ALLOWED = (                      # where the old name must remain, for compatibility or history
    "packages/compat/",          # the shim package
    "docs/plan/RENAME.md",
    "CHANGELOG.md",
)
COMPAT_LINES = re.compile(r"omnidiff\.(plugins|commands|eval/v1|reward/v1)|omnidiff-compat|main_deprecated")

files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split()
bad = []
for f in files:
    if f.startswith(ALLOWED) or f == "uv.lock":
        continue
    try:
        text = open(f, encoding="utf-8").read()
    except (UnicodeDecodeError, IsADirectoryError):
        continue
    for i, line in enumerate(text.splitlines(), 1):
        if OLD in line.lower() and not COMPAT_LINES.search(line):
            bad.append(f"{f}:{i}: {line.strip()[:100]}")
    if "/src/omnidiff/" in f and not f.startswith("packages/compat/"):
        bad.append(f"{f}: old namespace directory")

checks = [
    ["uv", "lock", "--check"],
    ["uv", "run", NEW_CLI, "--help"],
    ["uv", "run", ALIAS, "--help"],
    ["uv", "run", "python", "-c",
     f"import {NEW_PKG}.core, {NEW_PKG}.distill, {NEW_PKG}.reward, {NEW_PKG}.posttrain, {NEW_PKG}.data"],
    ["uv", "run", "python", "-c",
     f"from importlib.metadata import entry_points as e; "
     f"assert e(group='{NEW_PKG}.plugins') and e(group='{NEW_PKG}.commands')"],
    ["uv", "run", "pytest", "-q"],
]
for cmd in checks:
    if subprocess.run(cmd).returncode != 0:
        bad.append("failed: " + " ".join(cmd))

print("\n".join(bad) or "rename clean")
sys.exit(1 if bad else 0)
```

Run it with `uv run python scripts/check_rename.py`. The RN branch is ready for review when it prints `rename clean`.
