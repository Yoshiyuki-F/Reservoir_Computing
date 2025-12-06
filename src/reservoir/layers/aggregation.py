"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/layers/aggregation.py
Step 6 aggregation layers.
State aggregation components compatible with Transformer protocol.
"""

from __future__ import annotations

from typing import Dict, Any, Union

import jax.numpy as jnp
from reservoir.core.interfaces import Transformer
from reservoir.core.identifiers import AggregationMode

_ModeInput = Union[AggregationMode, str]


class StateAggregator(Transformer):
    """Stateless transformer that reduces the time axis using a configured mode."""

    def __init__(self, mode: _ModeInput = AggregationMode.LAST) -> None:
        self.mode = self._resolve_mode(mode)

    @staticmethod
    def _resolve_mode(mode: _ModeInput) -> AggregationMode:
        if isinstance(mode, AggregationMode):
            return mode
        if isinstance(mode, str):
            try:
                return AggregationMode(mode)
            except Exception as exc:
                raise ValueError(f"Invalid aggregation mode '{mode}'") from exc
        raise TypeError(f"Aggregation mode must be AggregationMode or str, got {type(mode)}.")

    @staticmethod
    def aggregate(states: jnp.ndarray, mode: _ModeInput) -> jnp.ndarray:
        """Static aggregator for reuse in functional contexts."""
        agg_mode = StateAggregator._resolve_mode(mode)
        arr = jnp.asarray(states, dtype=jnp.float64)
        if arr.ndim == 3:
            if agg_mode is AggregationMode.LAST:
                return arr[:, -1, :]
            if agg_mode is AggregationMode.MEAN:
                return jnp.mean(arr, axis=1)
            if agg_mode in {AggregationMode.LAST_MEAN, AggregationMode.MTS}:
                last = arr[:, -1, :]
                mean = jnp.mean(arr, axis=1)
                return jnp.concatenate([last, mean], axis=1)
            if agg_mode is AggregationMode.CONCAT:
                return arr.reshape(arr.shape[0], -1)
        elif arr.ndim == 2:
            if agg_mode is AggregationMode.LAST:
                return arr[-1]
            if agg_mode is AggregationMode.MEAN:
                return jnp.mean(arr, axis=0)
            if agg_mode in {AggregationMode.LAST_MEAN, AggregationMode.MTS}:
                last = arr[-1]
                mean = jnp.mean(arr, axis=0)
                return jnp.concatenate([last, mean], axis=0)
            if agg_mode is AggregationMode.CONCAT:
                return arr.reshape(-1)
        raise ValueError(f"Unsupported shape {arr.shape} or aggregation mode: {agg_mode}")

    def fit(self, features: jnp.ndarray) -> "StateAggregator":
        return self

    def transform(self, features: jnp.ndarray) -> jnp.ndarray:
        return StateAggregator.aggregate(features, self.mode)

    def fit_transform(self, features: jnp.ndarray) -> jnp.ndarray:
        return self.transform(features)

    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        return self.transform(features)

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode.value if isinstance(self.mode, AggregationMode) else str(self.mode)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateAggregator":
        return cls(mode=data.get("mode", AggregationMode.LAST))


__all__ = ["StateAggregator"]
