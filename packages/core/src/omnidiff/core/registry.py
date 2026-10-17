"""Name -> constructor registries. Every configurable component is built through one.

A spec is a string or a dict with a ``type`` key; the other keys are constructor
kwargs. ``type`` can also be ``"pkg.module:Name"`` to use something that was never
registered, which is the zero-boilerplate plugin path.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from typing import Any, Callable, Iterable


def import_object(path: str) -> Any:
    module, _, name = path.partition(":")
    obj = importlib.import_module(module)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._entries: dict[str, Callable[..., Any]] = {}

    def register(self, key: str | None = None):
        def deco(obj):
            k = key or obj.__name__
            existing = self._entries.get(k)
            if existing is not None and existing is not obj:
                raise KeyError(f"{self.name} '{k}' is already registered to {existing}")
            self._entries[k] = obj
            return obj

        return deco

    def get(self, key: str) -> Callable[..., Any]:
        if ":" in key:
            return import_object(key)
        try:
            return self._entries[key]
        except KeyError:
            raise KeyError(f"unknown {self.name} '{key}', available: {sorted(self._entries)}") from None

    def build(self, spec: dict | str, **context):
        """Build from a spec.

        ``context`` holds values shared across components (``dim``, ``sample_shape``,
        ``codec``...). Each one is passed only if the constructor accepts it and the
        spec did not set it, so components ask for what they need by naming it.
        """
        if spec is None:
            raise ValueError(f"{self.name} spec is None")
        if isinstance(spec, str):
            spec = {"type": spec}
        kwargs = dict(spec)
        ctor = self.get(kwargs.pop("type"))
        params = inspect.signature(ctor).parameters
        takes_any = any(p.kind is p.VAR_KEYWORD for p in params.values())
        for k, v in context.items():
            if k not in kwargs and (takes_any or k in params):
                kwargs[k] = v
        return ctor(**kwargs)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __iter__(self):
        return iter(sorted(self._entries))

    def __repr__(self) -> str:
        return f"Registry({self.name}: {', '.join(self)})"


MODELS = Registry("model")
ADAPTERS = Registry("adapter")
BACKBONES = Registry("backbone")
CONDITIONERS = Registry("conditioner")
PROCESSES = Registry("process")
CODECS = Registry("codec")
DATASETS = Registry("dataset")
PROMPTS = Registry("prompt source")
REWARDS = Registry("reward")
ALGORITHMS = Registry("algorithm")

ALL = {r.name: r for r in (MODELS, ADAPTERS, BACKBONES, CONDITIONERS, PROCESSES, CODECS, DATASETS, PROMPTS, REWARDS, ALGORITHMS)}


PLUGIN_GROUP = "omnidiff.plugins"
COMMAND_GROUP = "omnidiff.commands"


def entry_points(group: str):
    eps = importlib.metadata.entry_points()
    return eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])


_discovered = False


def discover_plugins() -> None:
    """Import every installed ``omnidiff.plugins`` entry point (each workspace package declares one).

    Installing a package is enough to make its components buildable from a config.
    """
    global _discovered
    if not _discovered:
        _discovered = True
        for ep in entry_points(PLUGIN_GROUP):
            ep.load()


def load_plugins(modules: Iterable[str] = ()) -> None:
    """Load installed plugin packages, then import ``modules`` so their ``@REGISTRY.register`` decorators run."""
    discover_plugins()
    for m in modules:
        importlib.import_module(m)
