"""
src/reservoir/components/preprocess/scaler.py
Feature scaling with persistent statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax.numpy as jnp
from reservoir.core.interfaces import Transformer


@dataclass
class ScalerState:
    mu: Optional[jnp.ndarray] = None
    sigma: Optional[jnp.ndarray] = None


class FeatureScaler(Transformer):
    """Simple z-score scaler with epsilon stabilization."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self.state = ScalerState()

    def fit(self, features: jnp.ndarray) -> "FeatureScaler":
        arr = jnp.asarray(features, dtype=jnp.float64)
        if arr.ndim not in (2, 3):
            raise ValueError(f"Feature matrix must be 2D or 3D, got shape {arr.shape}")
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        mu = jnp.mean(arr, axis=0)
        sigma = jnp.std(arr, axis=0) + self.eps
        self.state = ScalerState(mu=mu, sigma=sigma)
        return self

    def transform(self, features: jnp.ndarray) -> jnp.ndarray:
        if self.state.mu is None or self.state.sigma is None:
            raise RuntimeError("Scaler has not been fitted.")
        arr = jnp.asarray(features, dtype=jnp.float64)
        if arr.ndim not in (2, 3):
            raise ValueError(f"Feature matrix must be 2D or 3D, got shape {arr.shape}")
        if arr.ndim == 3:
            batch, time, feat = arr.shape
            arr = arr.reshape(-1, feat)
            transformed = (arr - self.state.mu) / self.state.sigma
            return transformed.reshape(batch, time, feat)
        return (arr - self.state.mu) / self.state.sigma

    def fit_transform(self, features: jnp.ndarray) -> jnp.ndarray:
        return self.fit(features).transform(features)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"eps": self.eps}
        if self.state.mu is not None and self.state.sigma is not None:
            data.update(
                {
                    "mu": jnp.asarray(self.state.mu).tolist(),
                    "sigma": jnp.asarray(self.state.sigma).tolist(),
                }
            )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureScaler":
        scaler = cls(eps=float(data.get("eps", 1e-8)))
        mu = data.get("mu")
        sigma = data.get("sigma")
        if mu is not None and sigma is not None:
            scaler.state = ScalerState(
                mu=jnp.asarray(mu, dtype=jnp.float64),
                sigma=jnp.asarray(sigma, dtype=jnp.float64),
            )
        return scaler
