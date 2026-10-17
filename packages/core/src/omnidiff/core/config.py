"""YAML configs with dotted command-line overrides (``rl.lr=1e-4``)."""

from __future__ import annotations

import copy
import dataclasses
import re
from typing import Iterable

import yaml


class _Loader(yaml.SafeLoader):
    pass


# PyYAML follows YAML 1.1 and reads `1e-4` as a string. Accept it as a float.
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def parse_value(raw: str):
    return yaml.load(raw, Loader=_Loader)


def load_config(path, overrides: Iterable[str] = ()) -> dict:
    with open(path) as f:
        cfg = yaml.load(f, Loader=_Loader) or {}
    return apply_overrides(cfg, overrides)


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[leaf] = parse_value(raw)
    return cfg


def save_config(cfg: dict, path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def from_dict(cls, data: dict | None):
    """Build a config dataclass, failing loudly on keys it does not know."""
    data = dict(data or {})
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown keys {sorted(unknown)}, expected some of {sorted(names)}")
    return cls(**data)


@dataclasses.dataclass
class SamplingConfig:
    num_steps: int = 20
    guidance_scale: float = 1.0
    timesteps: list[float] | None = None  # explicit descending grid from 1 to 0; overrides num_steps
