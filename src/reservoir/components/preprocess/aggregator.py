"""
src/reservoir/components/preprocess/aggregator.py
State aggregation components compatible with Transformer protocol.
"""

from __future__ import annotations

from typing import Literal, Dict, Any

import jax.numpy as jnp
from reservoir.core.interfaces import Transformer

AggregationMode = Literal["last", "mean", "last_mean", "mts", "concat"]


def aggregate_states(states: jnp.ndarray, mode: AggregationMode) -> jnp.ndarray:
    """
    Functional implementation of state aggregation (Legacy support & internal logic).
    """
    arr = jnp.asarray(states, dtype=jnp.float64)
    if arr.ndim == 3:
        if mode == "last":
            return arr[:, -1, :]
        if mode == "mean":
            return jnp.mean(arr, axis=1)
        if mode in {"last_mean", "mts"}:
            last = arr[:, -1, :]
            mean = jnp.mean(arr, axis=1)
            return jnp.concatenate([last, mean], axis=1)
        if mode == "concat":
            return arr.reshape(arr.shape[0], -1)
    elif arr.ndim == 2:
        if mode == "last":
            return arr[-1]
        if mode == "mean":
            return jnp.mean(arr, axis=0)
        if mode in {"last_mean", "mts"}:
            last = arr[-1]
            mean = jnp.mean(arr, axis=0)
            return jnp.concatenate([last, mean], axis=0)
        if mode == "concat":
            return arr.reshape(-1)
    raise ValueError(f"Unsupported shape {arr.shape} or aggregation mode: {mode}")


class StateAggregator(Transformer):
    """Stateless transformer that reduces the time axis using a configured mode."""

    def __init__(self, mode: AggregationMode = "last") -> None:
        self.mode = mode

    def fit(self, features: jnp.ndarray) -> "StateAggregator":
        return self

    def transform(self, features: jnp.ndarray) -> jnp.ndarray:
        return aggregate_states(features, self.mode)

    def fit_transform(self, features: jnp.ndarray) -> jnp.ndarray:
        return self.transform(features)

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateAggregator":
        return cls(mode=data.get("mode", "last"))
