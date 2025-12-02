"""
Input projection module shared across reservoir-based models.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp


class InputProjector:
    """Map raw inputs to reservoir space with optional sparsity and bias."""

    def __init__(
        self,
        n_inputs: int,
        n_units: int,
        input_scale: float,
        connectivity: float,
        bias_scale: float,
        seed: int,
    ) -> None:
        self.n_inputs = int(n_inputs)
        self.n_units = int(n_units)
        self.input_scale = float(input_scale)
        self.connectivity = float(connectivity)
        self.bias_scale = float(bias_scale)
        self.seed = int(seed)
        self._rng = jax.random.PRNGKey(self.seed)
        self._init_params()

    def _init_params(self) -> None:
        k_in, k_bias, k_mask_in = jax.random.split(self._rng, 3)
        self._rng = k_bias

        boundary = self.input_scale
        Win = jax.random.uniform(
            k_in,
            (self.n_inputs, self.n_units),
            minval=-boundary,
            maxval=boundary,
            dtype=jnp.float64,
        )

        if 0.0 < self.connectivity < 1.0:
            mask_in = jax.random.bernoulli(k_mask_in, p=self.connectivity, shape=Win.shape)
            Win = jnp.where(mask_in, Win, 0.0)

        bias = jax.random.uniform(
            k_bias,
            (self.n_units,),
            minval=-self.bias_scale,
            maxval=self.bias_scale,
            dtype=jnp.float64,
        )

        self.Win = Win
        self.bias = bias

    def project(self, inputs: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(inputs, dtype=jnp.float64)
        if arr.ndim != 3:
            raise ValueError(f"InputProjector expects (batch, time, features), got {arr.shape}")
        projected = jnp.einsum("bti,iu->btu", arr, self.Win)
        projected = projected + self.bias
        return projected

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return self.project(inputs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_inputs": self.n_inputs,
            "n_units": self.n_units,
            "input_scale": self.input_scale,
            "connectivity": self.connectivity,
            "bias_scale": self.bias_scale,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputProjector":
        return cls(
            n_inputs=int(data["n_inputs"]),
            n_units=int(data["n_units"]),
            input_scale=float(data["input_scale"]),
            connectivity=float(data["connectivity"]),
            bias_scale=float(data["bias_scale"]),
            seed=int(data["seed"]),
        )
