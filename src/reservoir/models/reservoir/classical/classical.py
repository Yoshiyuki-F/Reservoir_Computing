"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/reservoir/classical.py
Standard Echo State Network implementation.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp

from reservoir.core.identifiers import AggregationMode
from reservoir.layers.aggregation import StateAggregator
from reservoir.models.reservoir.base import Reservoir





class ClassicalReservoir(Reservoir):
    """Minimal ESN-style reservoir built on the Reservoir base class."""

    def __init__(
        self,
        n_units: int,
        spectral_radius: float,
        leak_rate: float,
        rc_connectivity: float,
        seed: int,
        aggregation_mode: AggregationMode,
    ) -> None:
        super().__init__(n_units=n_units, seed=seed)
        self.spectral_radius = float(spectral_radius)
        self.leak_rate = float(leak_rate)
        self.rc_connectivity = float(rc_connectivity)
        if not isinstance(aggregation_mode, AggregationMode):
            raise TypeError(f"aggregation_mode must be AggregationMode, got {type(aggregation_mode)}.")
        self.aggregator = StateAggregator(mode=aggregation_mode)
        self._rng = jax.random.PRNGKey(self.seed)
        self._init_weights()

    def _init_weights(self) -> None:
        k_res, k_mask_res = jax.random.split(self._rng, 2)
        self._rng = k_res

        W_dense = jax.random.normal(k_res, (self.n_units, self.n_units), dtype=jnp.float64)
        if 0.0 < self.rc_connectivity < 1.0:
            mask_res = jax.random.bernoulli(k_mask_res, p=self.rc_connectivity, shape=W_dense.shape)
            W_dense = jnp.where(mask_res, W_dense, 0.0)

        eig = jnp.max(jnp.abs(jnp.linalg.eigvals(W_dense)))
        scale = self.spectral_radius / eig if eig > 0 else 1.0
        self.W = W_dense * scale

    def initialize_state(self, batch_size: int = 1) -> jnp.ndarray:
        return jnp.zeros((batch_size, self.n_units), dtype=jnp.float64)

    def step(self, state: jnp.ndarray, projected_input: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # 論文の式:   (1 - a) * state + tanh(...)

        # 論文 Eq(7) 通りの実装 Li-ESN
        # next_state = (1.0 - self.leak_rate) * state + jnp.tanh(projected_input + jnp.dot(state, self.W))
        
        # Jaeger (2007) Standard Li-ESN
        next_state = (1.0 - self.leak_rate) * state + self.leak_rate * jnp.tanh(projected_input + jnp.dot(state, self.W))

        return next_state, next_state

    def forward(self, state: jnp.ndarray, input_data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if input_data.ndim != 3:
            raise ValueError(f"Expected batched sequences (batch, time, input), got {input_data.shape}")
        batch, time, feat = input_data.shape
        if feat != self.n_units:
            raise ValueError(f"Reservoir expects feature dim {self.n_units}, got {feat}")
        projected = input_data
        proj_transposed = jnp.swapaxes(projected, 0, 1)

        final_states, stacked = jax.lax.scan(self.step, state, proj_transposed)
        stacked = jnp.swapaxes(stacked, 0, 1)
        return final_states, stacked

    def __call__(self, inputs: jnp.ndarray, return_sequences: bool = False, split_name: str = None, **_: Any) -> jnp.ndarray:
        arr = jnp.asarray(inputs, dtype=jnp.float64)
        if arr.ndim != 3:
            raise ValueError(f"ClassicalReservoir expects 3D input (batch, time, features), got {arr.shape}")
        batch_size = arr.shape[0]
        initial_state = self.initialize_state(batch_size)
        _, artifacts = self.forward(initial_state, arr)
        states = artifacts

        # Zero-Overhead Logging (Step 5) via Callback - REMOVED due to XLA issues and redundancy
        # if split_name is not None:
        #      # Logic was causing INTERNAL: RET_CHECK failure
        #      pass

        if return_sequences:
            return states
        return self.aggregator.transform(states)

    def get_feature_dim(self, time_steps: int) -> int:
        """Return aggregated feature dimension without running the model."""
        return self.aggregator.get_output_dim(self.n_units, int(time_steps))

    def train(self, inputs: jnp.ndarray, targets: Any = None, **__: Any) -> Dict[str, Any]:
        """
        Reservoir has no trainable parameters; run forward for compatibility and return empty logs.
        """
        return {}

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "spectral_radius": self.spectral_radius,
                "leak_rate": self.leak_rate,
                "rc_connectivity": self.rc_connectivity,
                "seed": self.seed,
                "aggregation": self.aggregator.mode.value,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassicalReservoir":
        try:
            return cls(
                n_units=int(data["n_units"]),
                spectral_radius=float(data["spectral_radius"]),
                leak_rate=float(data["leak_rate"]),
                rc_connectivity=float(data["rc_connectivity"]),
                seed=int(data["seed"]),
                aggregation_mode=AggregationMode(data["aggregation"]),
            )
        except KeyError as exc:
            raise KeyError(f"Missing required reservoir parameter '{exc.args[0]}'") from exc

    def __repr__(self) -> str:
        return f"ClassicalReservoir(n_units={self.n_units})"
