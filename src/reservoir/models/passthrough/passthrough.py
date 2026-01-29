"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/passthrough/passthrough.py
Step 5 SKIP: Passthrough model that applies only aggregation (Step 6) to projected input.

Flow: [Batch, Time, Hidden] -> Aggregation -> [Batch, Feature]
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import jax
import jax.numpy as jnp

from reservoir.core.identifiers import AggregationMode
from reservoir.models.generative import ClosedLoopGenerativeModel
from reservoir.layers.aggregation import StateAggregator


class PassthroughModel(ClosedLoopGenerativeModel):
    """
    Model that skips dynamics (Step 5) and directly aggregates projected features.
    Implements the same interface as Reservoir for compatibility with GenericRunner.
    """

    def __init__(self, aggregation_mode: AggregationMode) -> None:
        if not isinstance(aggregation_mode, AggregationMode):
            raise TypeError(f"aggregation_mode must be AggregationMode, got {type(aggregation_mode)}.")
        self.aggregator = StateAggregator(mode=aggregation_mode)
        self.topology_meta: Dict[str, Any] = {}
        self._n_units: Optional[int] = None  # Set on first forward pass

    def train(self, inputs: jnp.ndarray, targets: Any = None, **_: Any) -> Dict[str, Any]:
        """No-op: Passthrough has no trainable parameters."""
        return {}

    # ------------------------------------------------------------------ #
    # Stateful Interface (Compatible with Reservoir)                     #
    # ------------------------------------------------------------------ #

    def initialize_state(self, batch_size: int = 1) -> jnp.ndarray:
        """Return zero state. Passthrough is stateless but needs compatible interface."""
        if self._n_units is None:
            raise RuntimeError("Passthrough n_units not set. Call forward() first.")
        return jnp.zeros((batch_size, self._n_units))

    def step(self, state: jnp.ndarray, projected_input: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Passthrough step: ignore state, return input as next state."""
        # projected_input: [batch, features] - ensure dtype matches state
        next_state = jnp.asarray(projected_input)
        return next_state, next_state

    def forward(self, state: jnp.ndarray, input_data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Process sequence. Returns (final_state, all_states)."""
        if input_data.ndim != 3:
            raise ValueError(f"Expected batched sequences (batch, time, input), got {input_data.shape}")
        
        # Use scan for consistency with Reservoir
        proj_transposed = jnp.swapaxes(input_data, 0, 1)  # [time, batch, feat]
        final_states, stacked = jax.lax.scan(self.step, state, proj_transposed)
        stacked = jnp.swapaxes(stacked, 0, 1)  # [batch, time, feat]
        return final_states, stacked
    
    # generate_closed_loop is inherited from ClosedLoopGenerativeModel

    # ------------------------------------------------------------------ #
    # Standard Interface                                                 #
    # ------------------------------------------------------------------ #

    def __call__(self, inputs: jnp.ndarray, split_name: str = None, **_: Any) -> jnp.ndarray:

        """Aggregate projected features. Accepts both 2D (Time, Features) and 3D (Batch, Time, Features). Output is 2D."""
        arr = jnp.asarray(inputs)
        
        # Convert 2D to 3D for internal processing
        if arr.ndim == 2:
            arr = arr[None, :, :]  # (T, F) -> (1, T, F)
        elif arr.ndim != 3:
            raise ValueError(f"PassthroughModel expects 2D or 3D input, got {arr.shape}")
        
        # Track n_units for initialize_state
        self._n_units = arr.shape[-1]
        
        # Aggregation always returns 2D
        log_label = f"6:{split_name}" if split_name else None
        return self.aggregator.transform(arr, log_label=log_label)

    def get_feature_dim(self, n_units: int, time_steps: int) -> int:
        """Return aggregated feature dimension."""
        return self.aggregator.get_output_dim(n_units, int(time_steps))

    def get_topology_meta(self) -> Dict[str, Any]:
        return self.topology_meta

    def __repr__(self) -> str:
        return f"PassthroughModel(agg={self.aggregator.mode.value})"
