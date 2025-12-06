"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/components/preprocess/scaler.py
Feature scaling transformer (standardization) compatible with the Transformer protocol.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import jax.numpy as jnp

from reservoir.core.interfaces import Transformer


class FeatureScaler(Transformer):
    """Simple standard scaler that centers and scales per feature."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = float(epsilon)
        self._mean: Optional[jnp.ndarray] = None
        self._std: Optional[jnp.ndarray] = None

    def fit(self, features: jnp.ndarray, y: Any = None) -> "FeatureScaler":
        arr = jnp.asarray(features, dtype=jnp.float64)
        if arr.ndim < 2:
            raise ValueError(f"FeatureScaler expects array with feature dimension, got shape {arr.shape}")
        self._mean = jnp.mean(arr, axis=0)
        self._std = jnp.std(arr, axis=0)
        self._std = jnp.where(self._std < self.epsilon, 1.0, self._std)
        return self

    def transform(self, features: jnp.ndarray) -> jnp.ndarray:
        if self._mean is None or self._std is None:
            # Lazy-fit to support transform-only usage in simple pipelines.
            self.fit(features)
        arr = jnp.asarray(features, dtype=jnp.float64)
        return (arr - self._mean) / self._std

    def fit_transform(self, features: jnp.ndarray) -> jnp.ndarray:
        return self.fit(features).transform(features)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"epsilon": self.epsilon}
        if self._mean is not None:
            data["mean"] = jnp.asarray(self._mean).tolist()
        if self._std is not None:
            data["std"] = jnp.asarray(self._std).tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureScaler":
        scaler = cls(epsilon=float(data.get("epsilon", 1e-8)))
        mean = data.get("mean")
        std = data.get("std")
        if mean is not None:
            scaler._mean = jnp.asarray(mean, dtype=jnp.float64)
        if std is not None:
            scaler._std = jnp.asarray(std, dtype=jnp.float64)
        return scaler


__all__ = ["FeatureScaler"]
