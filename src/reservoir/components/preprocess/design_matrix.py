"""
src/reservoir/components/preprocess/design_matrix.py
Design matrix expansion utilities (polynomial features + optional bias).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import chex
import jax.numpy as jnp

from reservoir.core.interfaces import Transformer


class DesignMatrix(Transformer):
    """Element-wise polynomial expansion with optional bias term."""

    def __init__(self, degree: int = 1, include_bias: bool = True, interaction_only: bool = False):
        self.degree = max(1, int(degree))
        self.include_bias = bool(include_bias)
        self.interaction_only = bool(interaction_only)
        self.input_dim: Optional[int] = None

    def fit(self, features: chex.Array, y: Optional[chex.Array] = None) -> "DesignMatrix":
        self.input_dim = features.shape[-1]
        return self

    def transform(self, features: chex.Array) -> chex.Array:
        parts = []
        if self.include_bias:
            bias_shape = features.shape[:-1] + (1,)
            parts.append(jnp.ones(bias_shape, dtype=features.dtype))
        parts.append(features)
        if self.degree > 1 and not self.interaction_only:
            for power in range(2, self.degree + 1):
                parts.append(jnp.power(features, power))
        return jnp.concatenate(parts, axis=-1)

    def fit_transform(self, features: chex.Array) -> chex.Array:
        return self.fit(features).transform(features)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "degree": self.degree,
            "include_bias": self.include_bias,
            "interaction_only": self.interaction_only,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DesignMatrix":
        return cls(
            degree=int(data.get("degree", 1)),
            include_bias=bool(data.get("include_bias", True)),
            interaction_only=bool(data.get("interaction_only", False)),
        )


__all__ = ["DesignMatrix"]
