"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/layers/projection.py
Step 3 Input projection module shared across reservoir-based models.
"""
from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp



class RandomProjection:
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
        bias_scale: float,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_scale = float(input_scale)
        self.connectivity = float(input_connectivity)
        self.bias_scale = float(bias_scale)
        self.seed = int(seed)
        k_w, k_b, k_mask = jax.random.split(jax.random.PRNGKey(self.seed), 3)

        boundary = self.input_scale
        W = jax.random.uniform(
            k_w,
            (self.input_dim, self.output_dim),
            minval=-boundary,
            maxval=boundary,
        )
        if 0.0 < self.connectivity < 1.0:
            mask = jax.random.bernoulli(k_mask, p=self.connectivity, shape=W.shape)
            W = jnp.where(mask, W, 0.0)
        bias = jnp.zeros((self.output_dim,))
        if self.use_bias:
            bias = jax.random.uniform(
                k_b,
                (self.output_dim,),
                minval=-self.bias_scale,
                maxval=self.bias_scale,
            )
        self.W = W
        self.bias = bias

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(inputs)
        if arr.ndim == 3:
            return jnp.einsum("bti,io->bto", arr, self.W) + self.bias
        if arr.ndim == 2:
            return jnp.dot(arr, self.W) + self.bias
        raise ValueError(f"RandomProjection expects 2D or 3D input, got shape {arr.shape}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_scale": self.input_scale,
            "connectivity": self.connectivity,
            "bias_scale": self.bias_scale,
            "seed": self.seed,
        }



class CenterCropProjection:
    """
    Fixed cropping projection that selects the central part of the input feature vector.
    Specific for image data where boundary pixels are often empty (e.g., MNIST).
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        if self.output_dim > self.input_dim:
            raise ValueError(
                f"CenterCropProjection: output_dim ({self.output_dim}) cannot be larger "
                f"than input_dim ({self.input_dim})."
            )

        # Calculate start index for centering
        # e.g., input=28, output=16 -> (28-16)//2 = 6. Take indices 6 to 22 (16 items).
        self.start_idx = (self.input_dim - self.output_dim) // 2
        self.end_idx = self.start_idx + self.output_dim

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # inputs shape: (Batch, Time, Features)
        if inputs.ndim != 3:
            raise ValueError(f"CenterCropProjection requires 3D input (Batch, Time, Features), got ndim={inputs.ndim}")
            
        # We assume the last dimension is the "Features" dimension to crop.
        return inputs[..., self.start_idx : self.end_idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "center_crop",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "crop_range": (self.start_idx, self.end_idx),
        }


__all__ = ["RandomProjection", "CenterCropProjection"]
