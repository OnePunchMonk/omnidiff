"""Framework-neutral student bundle: plain weights plus JSON that says how to sample them."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from omnidiff.core.config import load_config
from omnidiff.core.nn.lora import has_lora, merged_state_dict
from omnidiff.core.pipeline import Pipeline
from omnidiff.core.utils import CONFIG, load_weights

SCHEMA_VERSION = 1
SAMPLER_JSON, MODEL_JSON, README = "sampler.json", "model.json", "README.md"
SAFETENSORS, TORCH_WEIGHTS = "model.safetensors", "model.pt"
SAMPLER_REQUIRED = ("schema_version", "process", "sampler", "timesteps", "num_steps", "guidance_scale",
                    "sample_shape", "codec", "prompts")

_README = """# omnidiff student bundle

Weights: `{weights}` (a plain state dict, LoRA merged). `model.json` is the model spec and
`sampler.json` the sampling recipe (schema in docs/notes/D5.md).

Sample with omnidiff:

    from omnidiff.distill.export import load_bundle
    pipe = load_bundle("{name}")
    gen = pipe.generate(n=4, seed=0)

Without omnidiff: build the denoiser from `model.json`, load the weights, then integrate with the
sampler named in `sampler.json` over its `timesteps` (descending, 1 to 0) from noise of shape
`sample_shape`, at `guidance_scale`.
"""


def _safetensors():
    try:
        import safetensors.torch as st
    except ImportError:
        return None
    return st


def _plain(x):
    return json.loads(json.dumps(x))


def export_bundle(ckpt, out) -> Path:
    ckpt, out = Path(ckpt), Path(out)
    cfg = load_config(ckpt / CONFIG)
    pipe = Pipeline.from_checkpoint(ckpt, device="cpu")
    model = pipe.c.model
    if has_lora(model):
        state = merged_state_dict(model)
    else:
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state = {k: v.contiguous() for k, v in state.items()}

    process = copy.deepcopy(cfg["process"])
    if isinstance(process, str):
        process = {"type": process}
    sampler = process.pop("sampler", None)
    ptype = process.pop("type")
    sampling = pipe.c.sampling
    sampler_json = {
        "schema_version": SCHEMA_VERSION,
        "process": {"type": ptype, "settings": process},
        "sampler": sampler,
        "timesteps": list(sampling.timesteps) if sampling.timesteps is not None else None,
        "num_steps": sampling.num_steps,
        "guidance_scale": sampling.guidance_scale,
        "sample_shape": list(pipe.c.sample_shape),
        "codec": cfg.get("codec", "pixel"),
        "prompts": cfg.get("prompts", "none"),
    }
    if cfg.get("plugins"):
        sampler_json["plugins"] = list(cfg["plugins"])
    if "distilled_from" in cfg:
        sampler_json["distilled_from"] = cfg["distilled_from"]

    out.mkdir(parents=True, exist_ok=True)
    st = _safetensors()
    if st is not None:
        st.save_file(state, str(out / SAFETENSORS))
        weights = SAFETENSORS
    else:
        torch.save(state, out / TORCH_WEIGHTS)
        weights = TORCH_WEIGHTS
    (out / SAMPLER_JSON).write_text(json.dumps(_plain(sampler_json), indent=2) + "\n")
    (out / MODEL_JSON).write_text(json.dumps(_plain(cfg["model"]), indent=2) + "\n")
    (out / README).write_text(_README.format(weights=weights, name=out.name))
    return out


def load_bundle(path, device=None) -> Pipeline:
    path = Path(path)
    meta = json.loads((path / SAMPLER_JSON).read_text())
    missing = [k for k in SAMPLER_REQUIRED if k not in meta]
    if missing:
        raise KeyError(f"{SAMPLER_JSON}: missing keys {missing}")
    if meta["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported bundle schema_version {meta['schema_version']}, expected {SCHEMA_VERSION}")
    process = {"type": meta["process"]["type"], **meta["process"].get("settings", {})}
    if meta["sampler"] is not None:
        process["sampler"] = meta["sampler"]
    cfg = {
        "device": "cpu",
        "sample_shape": meta["sample_shape"],
        "codec": meta["codec"],
        "process": process,
        "model": json.loads((path / MODEL_JSON).read_text()),
        "prompts": meta["prompts"],
        "sampling": {"num_steps": meta["num_steps"], "guidance_scale": meta["guidance_scale"],
                     "timesteps": meta["timesteps"]},
    }
    if "plugins" in meta:
        cfg["plugins"] = meta["plugins"]
    pipe = Pipeline.from_config(cfg, device)
    if (path / SAFETENSORS).exists():
        st = _safetensors()
        if st is None:
            raise ImportError(f"{SAFETENSORS} needs safetensors: pip install omnidiff-distill[export]")
        state = st.load_file(str(path / SAFETENSORS), device=str(pipe.c.device))
    else:
        state = load_weights(path / TORCH_WEIGHTS, pipe.c.device)
    pipe.c.model.load_state_dict(state)
    return pipe
