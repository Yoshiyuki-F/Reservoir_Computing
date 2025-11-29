"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/reservoir/classical.py
Standard Echo State Network implementation.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp

from reservoir.models.reservoir.base import Reservoir

StepArtifacts = namedtuple("StepArtifacts", ["states"])


class ClassicalReservoir(Reservoir):
    """Minimal ESN-style reservoir implementing the ReservoirNode protocol."""

    def __init__(
        self,
        n_inputs: int,
        n_units: int,
        input_scale: float = 1.0,
        spectral_radius: float = 0.95,
        leak_rate: float = 1.0,
        connectivity: float = 0.1,
        noise_rc: float = 0.0,
        bias_scale: float = 0.0,
        seed: int = 42,
    ) -> None:
        super().__init__(n_inputs=n_inputs, n_units=n_units, noise_rc=noise_rc)
        self.input_scale = float(input_scale)
        self.spectral_radius = float(spectral_radius)
        self.leak_rate = float(leak_rate)
        self.connectivity = float(connectivity)
        self.bias_scale = float(bias_scale)
        self.seed = int(seed)
        self._rng = jax.random.PRNGKey(self.seed)
        self._init_weights()

    def _init_weights(self) -> None:
        k_in, k_res, k_bias = jax.random.split(self._rng, 3)
        self._rng = k_bias
        boundary = self.input_scale / jnp.sqrt(self.n_inputs)
        self.Win = jax.random.uniform(
            k_in,
            (self.n_inputs, self.n_units),
            minval=-boundary,
            maxval=boundary,
            dtype=jnp.float64,
        )
        W_dense = jax.random.normal(k_res, (self.n_units, self.n_units), dtype=jnp.float64)
        if 0.0 < self.connectivity < 1.0:
            mask = jax.random.bernoulli(k_res, p=self.connectivity, shape=W_dense.shape)
            W_dense = jnp.where(mask, W_dense, 0.0)
        eig = jnp.max(jnp.abs(jnp.linalg.eigvals(W_dense)))
        scale = self.spectral_radius / eig if eig > 0 else 1.0
        self.W = W_dense * scale
        self.bias = jax.random.uniform(
            k_bias,
            (self.n_units,),
            minval=-self.bias_scale,
            maxval=self.bias_scale,
            dtype=jnp.float64,
        )

    def initialize_state(self, batch_size: int = 1) -> jnp.ndarray:
        return jnp.zeros((batch_size, self.n_units), dtype=jnp.float64)

    def step(self, state: jnp.ndarray, input_data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pre_activation = jnp.dot(input_data, self.Win) + jnp.dot(state, self.W) + self.bias
        activated = jnp.tanh(pre_activation)
        next_state = (1.0 - self.leak_rate) * state + self.leak_rate * activated
        return next_state, next_state

    def forward(self, state: jnp.ndarray, input_data: jnp.ndarray) -> Tuple[jnp.ndarray, StepArtifacts]:
        if input_data.ndim != 3:
            raise ValueError(f"Expected batched sequences (batch, time, input), got {input_data.shape}")
        batch, time, _ = input_data.shape
        inputs_transposed = jnp.swapaxes(input_data, 0, 1)

        def scan_fn(carry, x_t):
            return self.step(carry, x_t)

        final_states, stacked = jax.lax.scan(scan_fn, state, inputs_transposed)
        stacked = jnp.swapaxes(stacked, 0, 1)
        return final_states, StepArtifacts(states=stacked)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "input_scale": self.input_scale,
                "spectral_radius": self.spectral_radius,
                "leak_rate": self.leak_rate,
                "connectivity": self.connectivity,
                "bias_scale": self.bias_scale,
                "seed": self.seed,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassicalReservoir":
        return cls(
            n_inputs=int(data["n_inputs"]),
            n_units=int(data["n_units"]),
            input_scale=float(data.get("input_scale", 1.0)),
            spectral_radius=float(data.get("spectral_radius", 0.95)),
            leak_rate=float(data.get("leak_rate", 1.0)),
            connectivity=float(data.get("connectivity", 0.1)),
            noise_rc=float(data.get("noise_rc", 0.0)),
            bias_scale=float(data.get("bias_scale", 0.0)),
            seed=int(data.get("seed", 42)),
        )
