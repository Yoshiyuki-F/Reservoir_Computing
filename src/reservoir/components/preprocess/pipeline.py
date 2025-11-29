"""
src/reservoir/components/preprocess/pipeline.py
Sequential composition of Transformer components.
"""
from __future__ import annotations

from importlib import import_module
from typing import Iterable, List, Dict, Any

import jax.numpy as jnp

from reservoir.core.interfaces import Transformer


class TransformerSequence(Transformer):
    """Applies a list of Transformer instances in order."""

    def __init__(self, transformers: Iterable[Transformer]) -> None:
        self.transformers: List[Transformer] = [t for t in transformers if t is not None]

    def fit(self, features: jnp.ndarray) -> "TransformerSequence":
        data = features
        for transformer in self.transformers:
            data = transformer.fit_transform(data)
        return self

    def transform(self, features: jnp.ndarray) -> jnp.ndarray:
        data = features
        for transformer in self.transformers:
            data = transformer.transform(data)
        return data

    def fit_transform(self, features: jnp.ndarray) -> jnp.ndarray:
        data = features
        for transformer in self.transformers:
            data = transformer.fit_transform(data)
        return data

    def to_dict(self) -> Dict[str, Any]:
        serialized = []
        for transformer in self.transformers:
            cls = transformer.__class__
            path = f"{cls.__module__}.{cls.__qualname__}"
            config = transformer.to_dict()
            serialized.append({"class": path, "config": config})
        return {"transformers": serialized}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformerSequence":
        transformers: List[Transformer] = []
        for entry in data.get("transformers", []):
            path = entry["class"]
            module_name, _, class_name = path.rpartition(".")
            module = import_module(module_name)
            transformer_cls = getattr(module, class_name)
            config = entry.get("config", {})
            if hasattr(transformer_cls, "from_dict"):
                instance = transformer_cls.from_dict(config)
            else:
                instance = transformer_cls(**config)
            transformers.append(instance)
        return cls(transformers)
