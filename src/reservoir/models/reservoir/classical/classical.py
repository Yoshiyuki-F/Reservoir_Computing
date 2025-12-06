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
    """Minimal ESN-style reservoir built on the Reservoir base class."""

    def __init__(
        self,
        n_inputs: int,
        n_units: int,
        spectral_radius: float,
        leak_rate: float,
        rc_connectivity: float,
        noise_rc: float,
        seed: int,
    ) -> None:
        super().__init__(n_inputs=n_inputs, n_units=n_units, noise_rc=noise_rc, seed=seed)
        self.spectral_radius = float(spectral_radius)
        self.leak_rate = float(leak_rate)
        self.rc_connectivity = float(rc_connectivity)
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

        pre_activation = projected_input + jnp.dot(state, self.W)
        activated = jnp.tanh(pre_activation)

        # 論文 Eq(7) 通りの実装
        next_state = (1.0 - self.leak_rate) * state + activated

        return next_state, next_state

    def forward(self, state: jnp.ndarray, input_data: jnp.ndarray) -> Tuple[jnp.ndarray, StepArtifacts]:
        if input_data.ndim != 3:
            raise ValueError(f"Expected batched sequences (batch, time, input), got {input_data.shape}")
        batch, time, feat = input_data.shape
        if feat != self.n_units:
            raise ValueError(f"Reservoir expects feature dim {self.n_units}, got {feat}")
        projected = input_data
        proj_transposed = jnp.swapaxes(projected, 0, 1)

        final_states, stacked = jax.lax.scan(self.step, state, proj_transposed)
        stacked = jnp.swapaxes(stacked, 0, 1)
        return final_states, StepArtifacts(states=stacked)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "spectral_radius": self.spectral_radius,
                "leak_rate": self.leak_rate,
                "rc_connectivity": self.rc_connectivity,
                "seed": self.seed,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassicalReservoir":
        try:
            return cls(
                n_inputs=int(data["n_inputs"]),
                n_units=int(data["n_units"]),
                spectral_radius=float(data["spectral_radius"]),
                leak_rate=float(data["leak_rate"]),
                rc_connectivity=float(data["rc_connectivity"]),
                noise_rc=float(data["noise_rc"]),
                seed=int(data["seed"]),
            )
        except KeyError as exc:
            raise KeyError(f"Missing required reservoir parameter '{exc.args[0]}'") from exc

    def __repr__(self) -> str:
        return f"ClassicalReservoir(n_inputs={self.n_inputs}, n_units={self.n_units})"
