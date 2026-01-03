"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/passthrough/passthrough.py
Step 5 SKIP: Passthrough model that applies only aggregation (Step 6) to projected input.

Flow: [Batch, Time, Hidden] -> Aggregation -> [Batch, Feature]
"""
from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from reservoir.core.identifiers import AggregationMode
from reservoir.layers.aggregation import StateAggregator


class PassthroughModel:
    """
    Model that skips dynamics (Step 5) and directly aggregates projected features.
    Only needs aggregation mode - derives everything else from input.
    """

    def __init__(self, aggregation_mode: AggregationMode) -> None:
        if not isinstance(aggregation_mode, AggregationMode):
            raise TypeError(f"aggregation_mode must be AggregationMode, got {type(aggregation_mode)}.")
        self.aggregator = StateAggregator(mode=aggregation_mode)
        self.topology_meta: Dict[str, Any] = {}

    def train(self, inputs: jnp.ndarray, targets: Any = None, **_: Any) -> Dict[str, Any]:
        """No-op: Passthrough has no trainable parameters."""
        return {}

    def __call__(self, inputs: jnp.ndarray, **_: Any) -> jnp.ndarray:
        """Aggregate projected features directly. Input: [B, T, H] -> Output: [B, F]"""
        arr = jnp.asarray(inputs, dtype=jnp.float64)
        if arr.ndim != 3:
            raise ValueError(f"PassthroughModel expects 3D input (batch, time, features), got {arr.shape}")
        return self.aggregator.transform(arr)

    def get_feature_dim(self, n_units: int, time_steps: int) -> int:
        """Return aggregated feature dimension."""
        return self.aggregator.get_output_dim(n_units, int(time_steps))

    def get_topology_meta(self) -> Dict[str, Any]:
        return self.topology_meta

    def __repr__(self) -> str:
        return f"PassthroughModel(agg={self.aggregator.mode.value})"
