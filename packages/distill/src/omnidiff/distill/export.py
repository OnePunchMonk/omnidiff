"""Framework-neutral student bundle: plain weights plus JSON that says how to sample them."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import torch

from omnidiff.core.config import load_config
from omnidiff.core.pipeline import Pipeline
from omnidiff.core.utils import CONFIG, load_weights

SCHEMA_VERSION = 1
SAMPLER_JSON, MODEL_JSON, README = "sampler.json", "model.json", "README.md"
SAFETENSORS, TORCH_WEIGHTS = "model.safetensors", "model.pt"
RESOURCES = "resources"
SAMPLER_REQUIRED = ("schema_version", "process", "sampler", "timesteps", "num_steps", "guidance_scale",
                    "sample_shape", "codec", "prompts", "weights")

_README = """# omnidiff student bundle

Files: `{weights}` (plain state dict, LoRA merged), `model.json` (model spec, the `model` section of an
omnidiff config), `sampler.json` (below), `resources/` (files named by codec or prompt specs).
Loading a bundle imports any plugin modules it names under `plugins`, so only load bundles you trust.

    from omnidiff.distill.export import load_bundle
    pipe = load_bundle("{name}")
    gen = pipe.generate(n=4, seed=0)

## sampler.json (schema_version 1)

Required: `schema_version` (1); `process` (`{{type, settings}}`, registry name and constructor settings);
`sampler` (str or null: renoise, ode, coupled, ...); `timesteps` (strictly descending floats from 1 to 0, or
null to use the process grid with `num_steps`); `num_steps` (equals len(timesteps) - 1 when both are given);
`guidance_scale`; `sample_shape` (per-sample shape in process space); `codec` and `prompts` (registry specs;
relative `path` entries are relative to the bundle); `weights` (`{{file, format, sha256}}`).
Optional: `distilled_from` (teacher, method, teacher_guidance, steps), `plugins`.

## Sampling semantics (omnidiff.core.sampling)

Time runs from t=1 (pure noise / all mask) to t=0 (data). The model is called as `model(x, t, cond)`; a
`two_time` model (see model.json) is called as `model(x, t, cond, s=...)` where s is the target time of the
jump: the next grid time for ode/coupled, 0 for renoise (jump to data, then re-noise to the next time).
Prediction target: `flow` velocity (noise - x0), `gaussian` epsilon, `masked` logits over clean tokens.
Prior: Gaussian noise for flow/gaussian; all-mask tokens for masked. The `coupled` sampler also needs
`cond["unmask_time"]` (B, L) and `cond["token_noise"]` (B, L) uniform or (B, L, V) Gumbel, drawn by
`process.init_noise`; students with a noise-conditioned token adapter read `token_noise` themselves.
Guidance (when the model supports it, scale != 1): `uncond + scale * (cond - uncond)` on the model output,
with the condition dropped for the uncond branch. Students are normally exported with scale 1.
"""


def _safetensors():
    try:
        import safetensors.torch as st
    except ImportError:
        return None
    return st


def _plain(x):
    return json.loads(json.dumps(x))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _map_paths(spec, fn):
    """Apply ``fn`` to every string under a ``path`` / ``*_path`` key of a nested spec."""
    if isinstance(spec, dict):
        return {k: fn(v) if isinstance(v, str) and (k == "path" or k.endswith("_path")) else _map_paths(v, fn)
                for k, v in spec.items()}
    if isinstance(spec, list):
        return [_map_paths(v, fn) for v in spec]
    return spec


def _teacher_name(teacher: str) -> str:
    """Keep the checkpoint's identity without the absolute local prefix."""
    p = Path(teacher)
    return "/".join(p.parts[-2:]) if p.is_absolute() else teacher


def validate_sampler(meta: dict) -> None:
    missing = [k for k in SAMPLER_REQUIRED if k not in meta]
    if missing:
        raise ValueError(f"{SAMPLER_JSON}: missing keys {missing}")
    if meta["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported bundle schema_version {meta['schema_version']}, expected {SCHEMA_VERSION}")
    proc = meta["process"]
    if not isinstance(proc, dict) or not isinstance(proc.get("type"), str):
        raise ValueError(f"{SAMPLER_JSON}: process must be an object with a string `type`, got {proc!r}")
    if not isinstance(proc.get("settings", {}), dict):
        raise ValueError(f"{SAMPLER_JSON}: process.settings must be an object")
    ts = meta["timesteps"]
    if ts is not None:
        numeric = isinstance(ts, list) and len(ts) >= 2 and all(
            isinstance(t, (int, float)) and not isinstance(t, bool) for t in ts)
        if not numeric or ts[0] != 1 or ts[-1] != 0 or any(b >= a for a, b in zip(ts, ts[1:])):
            raise ValueError(f"{SAMPLER_JSON}: timesteps must be strictly descending numbers from 1 to 0, got {ts}")
        if meta["num_steps"] != len(ts) - 1:
            raise ValueError(f"{SAMPLER_JSON}: num_steps {meta['num_steps']} != len(timesteps) - 1 = {len(ts) - 1}")
    shape = meta["sample_shape"]
    if not isinstance(shape, list) or not shape or not all(isinstance(d, int) and d > 0 for d in shape):
        raise ValueError(f"{SAMPLER_JSON}: sample_shape must be a list of positive ints, got {shape!r}")
    w = meta["weights"]
    if not isinstance(w, dict) or w.get("format") not in ("safetensors", "torch") or not {"file", "sha256"} <= set(w):
        raise ValueError(f"{SAMPLER_JSON}: weights must be {{file, format, sha256}} with format safetensors or torch")


def export_bundle(ckpt, out) -> Path:
    ckpt, out = Path(ckpt), Path(out)
    cfg = load_config(ckpt / CONFIG)
    pipe = Pipeline.from_checkpoint(ckpt, device="cpu")
    # Pipeline builds the plain architecture (no LoRA wrappers) and checkpoints already store merged
    # weights, so the model's state dict is the portable one.
    state = {k: v.detach().clone().contiguous() for k, v in pipe.c.model.state_dict().items()}

    process = copy.deepcopy(cfg["process"])
    if isinstance(process, str):
        process = {"type": process}
    sampler = process.pop("sampler", None)
    ptype = process.pop("type")
    sampling = pipe.c.sampling

    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out / RESOURCES, ignore_errors=True)
    copied = {}

    def bring(path: str) -> str:
        if path not in copied:
            src = Path(path)
            dst = f"{RESOURCES}/{len(copied)}_{src.name}"
            (out / RESOURCES).mkdir(exist_ok=True)
            shutil.copyfile(src, out / dst)
            copied[path] = dst
        return copied[path]

    sampler_json = {
        "schema_version": SCHEMA_VERSION,
        "process": {"type": ptype, "settings": process},
        "sampler": sampler,
        "timesteps": list(sampling.timesteps) if sampling.timesteps is not None else None,
        "num_steps": sampling.num_steps,
        "guidance_scale": sampling.guidance_scale,
        "sample_shape": list(pipe.c.sample_shape),
        "codec": _map_paths(cfg.get("codec", "pixel"), bring),
        "prompts": _map_paths(cfg.get("prompts", "none"), bring),
    }
    if cfg.get("plugins"):
        sampler_json["plugins"] = list(cfg["plugins"])
    if "distilled_from" in cfg:
        df = dict(cfg["distilled_from"])
        if "teacher" in df:
            df["teacher"] = _teacher_name(str(df["teacher"]))
        sampler_json["distilled_from"] = df

    # Exactly one weights file per bundle: a re-export must not leave the other format behind.
    for name in (SAFETENSORS, TORCH_WEIGHTS):
        (out / name).unlink(missing_ok=True)
    st = _safetensors()
    if st is not None:
        weights, fmt = SAFETENSORS, "safetensors"
        st.save_file(state, str(out / weights))
    else:
        weights, fmt = TORCH_WEIGHTS, "torch"
        torch.save(state, out / weights)
    sampler_json["weights"] = {"file": weights, "format": fmt, "sha256": _sha256(out / weights)}
    (out / SAMPLER_JSON).write_text(json.dumps(_plain(sampler_json), indent=2) + "\n")
    (out / MODEL_JSON).write_text(json.dumps(_plain(cfg["model"]), indent=2) + "\n")
    (out / README).write_text(_README.format(weights=weights, name=out.name))
    return out


def load_bundle(path, device=None) -> Pipeline:
    """Build a Pipeline from a bundle. Imports the plugin modules named in ``sampler.json``: trust the bundle."""
    path = Path(path)
    meta = json.loads((path / SAMPLER_JSON).read_text())
    validate_sampler(meta)
    w = meta["weights"]
    wfile = path / w["file"]
    if Path(w["file"]).name != w["file"]:
        raise ValueError(f"{SAMPLER_JSON}: weights file must sit in the bundle root, got {w['file']!r}")
    if _sha256(wfile) != w["sha256"]:
        raise ValueError(f"{wfile} does not match the sha256 recorded in {SAMPLER_JSON}")

    def local(p: str) -> str:
        return p if Path(p).is_absolute() else str(path / p)

    process = {"type": meta["process"]["type"], **meta["process"].get("settings", {})}
    if meta["sampler"] is not None:
        process["sampler"] = meta["sampler"]
    cfg = {
        "device": "cpu",
        "sample_shape": meta["sample_shape"],
        "codec": _map_paths(meta["codec"], local),
        "process": process,
        "model": json.loads((path / MODEL_JSON).read_text()),
        "prompts": _map_paths(meta["prompts"], local),
        "sampling": {"num_steps": meta["num_steps"], "guidance_scale": meta["guidance_scale"],
                     "timesteps": meta["timesteps"]},
    }
    if "plugins" in meta:
        cfg["plugins"] = meta["plugins"]
    pipe = Pipeline.from_config(cfg, device)
    if w["format"] == "safetensors":
        st = _safetensors()
        if st is None:
            raise ImportError(f"{w['file']} needs safetensors: pip install omnidiff-distill[export]")
        state = st.load_file(str(wfile), device="cpu")
    else:
        state = load_weights(wfile, "cpu")
    pipe.c.model.load_state_dict(state)
    return pipe
