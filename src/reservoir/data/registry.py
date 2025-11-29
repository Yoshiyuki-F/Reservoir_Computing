"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/registry.py
Registry for dataset loader functions."""

from __future__ import annotations

from typing import Callable, Dict, Tuple, Any

import jax.numpy as jnp

# Dataset loader signature: takes config dict, returns (X, y)
DatasetLoaderFn = Callable[[Dict[str, Any]], Tuple[jnp.ndarray, jnp.ndarray]]


class DatasetRegistry:
    """Simple name → loader registry for datasets."""

    _REGISTRY: Dict[str, DatasetLoaderFn] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[DatasetLoaderFn], DatasetLoaderFn]:
        """Decorator to register a dataset loader."""

        def decorator(fn: DatasetLoaderFn) -> DatasetLoaderFn:
            cls._REGISTRY[name.lower()] = fn
            return fn

        return decorator

    @classmethod
    def get(cls, name: str) -> DatasetLoaderFn:
        """Fetch a registered loader, resolving simple aliases."""
        key = name.lower().replace("-", "_")
        # Alias handling
        if key in {"sine", "sw"}:
            key = "sine_wave"
        elif key == "m":
            key = "mnist"

        if key not in cls._REGISTRY:
            available = ", ".join(sorted(cls._REGISTRY.keys()))
            raise ValueError(f"Dataset '{name}' not found. Available: {available}")
        return cls._REGISTRY[key]
