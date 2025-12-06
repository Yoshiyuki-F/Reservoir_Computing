"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/layers/projection.py
Step 3 Input projection module shared across reservoir-based models.
"""
from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp

class InputProjection:
    """
    Fixed random input projection with sparsity and bias (separable from reservoir dynamics).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        input_scale: float,
        input_connectivity: float,
        seed: int,
        use_bias: bool = True,
        bias_scale: float = 0.0,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_scale = float(input_scale)
        self.connectivity = float(input_connectivity)
        self.use_bias = bool(use_bias)
        self.bias_scale = float(bias_scale)
        self.seed = int(seed)
        k_w, k_b, k_mask = jax.random.split(jax.random.PRNGKey(self.seed), 3)

        boundary = self.input_scale
        W = jax.random.uniform(
            k_w,
            (self.input_dim, self.output_dim),
            minval=-boundary,
            maxval=boundary,
            dtype=jnp.float64,
        )
        if 0.0 < self.connectivity < 1.0:
            mask = jax.random.bernoulli(k_mask, p=self.connectivity, shape=W.shape)
            W = jnp.where(mask, W, 0.0)
        bias = jnp.zeros((self.output_dim,), dtype=jnp.float64)
        if self.use_bias:
            bias = jax.random.uniform(
                k_b,
                (self.output_dim,),
                minval=-self.bias_scale,
                maxval=self.bias_scale,
                dtype=jnp.float64,
            )
        self.W = W
        self.bias = bias

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(inputs, dtype=jnp.float64)
        if arr.ndim == 3:
            return jnp.einsum("bti,io->bto", arr, self.W) + self.bias
        if arr.ndim == 2:
            return jnp.dot(arr, self.W) + self.bias
        raise ValueError(f"InputProjection expects 2D or 3D input, got shape {arr.shape}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_scale": self.input_scale,
            "connectivity": self.connectivity,
            "bias_scale": self.bias_scale,
            "seed": self.seed,
        }


__all__ = ["InputProjection"]
