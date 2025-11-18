"""Design matrix expansion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import jax.numpy as jnp

PolyMode = Literal["none", "square", "poly_k"]


@dataclass
class DesignState:
    keep_mask: Optional[jnp.ndarray] = None


class DesignMatrixBuilder:
    """Constructs expanded feature matrices with polynomial terms and bias."""

    def __init__(
        self,
        *,
        poly_mode: PolyMode = "square",
        degree: int = 2,
        include_bias: bool = True,
        std_threshold: float = 1e-3,
    ) -> None:
        self.poly_mode = poly_mode
        self.degree = max(1, int(degree))
        self.include_bias = include_bias
        self.std_threshold = float(std_threshold)
        self.state = DesignState()

    def fit(self, features) -> "DesignMatrixBuilder":
        arr = jnp.asarray(features, dtype=jnp.float64)
        if arr.ndim != 2:
            raise ValueError(f"Design builder expects 2D input, got {arr.shape}")
        if arr.shape[1] == 0:
            self.state.keep_mask = jnp.ones((0,), dtype=bool)
            return self
        std = jnp.std(arr, axis=0)
        keep = std > self.std_threshold
        if not jnp.any(keep):
            keep = jnp.ones_like(keep, dtype=bool)
        self.state.keep_mask = keep
        return self

    def transform(self, features):
        if self.state.keep_mask is None:
            raise RuntimeError("DesignMatrixBuilder has not been fitted.")
        arr = jnp.asarray(features, dtype=jnp.float64)
        arr = arr[:, self.state.keep_mask]

        expanded = self._apply_polynomial_expansion(arr)
        if self.include_bias:
            bias = jnp.ones((expanded.shape[0], 1), dtype=expanded.dtype)
            expanded = jnp.concatenate([expanded, bias], axis=1)
        return expanded

    def fit_transform(self, features):
        return self.fit(features).transform(features)

    def _apply_polynomial_expansion(self, features: jnp.ndarray) -> jnp.ndarray:
        if self.poly_mode == "none":
            return features
        if self.poly_mode == "square":
            return jnp.concatenate([features, features**2], axis=1)
        if self.poly_mode == "poly_k":
            polys = [features]
            for k in range(2, self.degree + 1):
                polys.append(features**k)
            return jnp.concatenate(polys, axis=1)
        raise ValueError(f"Unknown poly_mode '{self.poly_mode}'")

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "poly_mode": self.poly_mode,
            "degree": self.degree,
            "include_bias": self.include_bias,
            "std_threshold": self.std_threshold,
        }
        if self.state.keep_mask is not None:
            data["keep_mask"] = jnp.asarray(self.state.keep_mask).tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DesignMatrixBuilder":
        builder = cls(
            poly_mode=data.get("poly_mode", "square"),
            degree=int(data.get("degree", 2)),
            include_bias=bool(data.get("include_bias", True)),
            std_threshold=float(data.get("std_threshold", 1e-3)),
        )
        keep = data.get("keep_mask")
        if keep is not None:
            builder.state.keep_mask = jnp.asarray(keep, dtype=bool)
        return builder
