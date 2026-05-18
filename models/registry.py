"""
Model and condition-encoder registries.

Usage:
    @MODEL_REGISTRY.register("WanWorldModel")
    class WanWorldModel(nn.Module): ...

    model = MODEL_REGISTRY.build(cfg)   # cfg.model.name == "WanWorldModel"

    @CONDITION_REGISTRY.register("text")
    class TextConditionEncoder(ConditionEncoder): ...

    encoder = CONDITION_REGISTRY.build(cfg.model.condition)
"""

from __future__ import annotations

from typing import Any, Dict, Type

from utils.utils import cfg_get


class _Registry:
    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: str):
        def decorator(cls):
            if name in self._registry and self._registry[name] is not cls:
                raise ValueError(
                    f"[{self._name}] '{name}' already registered by {self._registry[name].__name__}"
                )
            self._registry[name] = cls
            return cls
        return decorator

    def build(self, cfg: Any, **kwargs):
        name = cfg_get(cfg, "name", None)
        if name is None:
            raise ValueError(f"[{self._name}] config must have a 'name' field")
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"[{self._name}] '{name}' not found. Available: [{available}]"
            )
        return self._registry[name](cfg, **kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __getitem__(self, name: str) -> Type:
        return self._registry[name]

    def keys(self):
        return self._registry.keys()


MODEL_REGISTRY = _Registry("ModelRegistry")
CONDITION_REGISTRY = _Registry("ConditionRegistry")
