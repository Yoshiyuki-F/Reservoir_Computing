"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/layers/projection.py
STEP3 Projection which changes the dimension of feature vectors.
"""
from __future__ import annotations
import abc
from typing import Any, Dict, Type
from functools import singledispatch

import jax
import jax.numpy as jnp

# --- 1. Interface Definition ---

class Projection(abc.ABC):
    """
    Abstract Base Class for Input Projections.
    Defines the contract that all projection layers must follow.
    Also handles common state (input_dim, output_dim).
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim = int(input_dim)
        self._output_dim = int(output_dim)
    
    @property
    def output_dim(self) -> int:
        """Return the output dimension size."""
        return self._output_dim

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the projection to the inputs.
        template method: validates input shape then calls _forward.
        """
        arr = jnp.asarray(inputs)
        if arr.ndim not in (2, 3):
             raise ValueError(f"{self.__class__.__name__} expects 2D or 3D input, got shape {arr.shape}")
        return self._project(arr)

    @abc.abstractmethod
    def _project(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Actual projection logic to be implemented by subclasses."""
        pass

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration parameters."""
        pass


# --- 2. Concrete Implementations ---

class RandomProjection(Projection):
    """
    Fixed random input projection with sparsity and bias.
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
        super().__init__(input_dim, output_dim)
        self.input_scale = float(input_scale)
        self.connectivity = float(input_connectivity)
        self.bias_scale = float(bias_scale)
        self.seed = int(seed)
        
        # Determine if bias is used based on scale
        self.use_bias = self.bias_scale > 0.0

        k_w, k_b, k_mask = jax.random.split(jax.random.PRNGKey(self.seed), 3)

        boundary = self.input_scale
        W = jax.random.uniform(
            k_w,
            (self.input_dim, self._output_dim),
            minval=-boundary,
            maxval=boundary,
        )
        if 0.0 < self.connectivity < 1.0:
            mask = jax.random.bernoulli(k_mask, p=self.connectivity, shape=W.shape)
            W = jnp.where(mask, W, 0.0)
        
        bias = jnp.zeros((self._output_dim,))
        if self.use_bias:
            bias = jax.random.uniform(
                k_b,
                (self._output_dim,),
                minval=-self.bias_scale,
                maxval=self.bias_scale,
            )
        self.W = W
        self.bias = bias

    def _project(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # Template ensures inputs is 2D or 3D ndarray
        if inputs.ndim == 3:
            return jnp.einsum("bti,io->bto", inputs, self.W) + self.bias
        # inputs.ndim == 2
        return jnp.dot(inputs, self.W) + self.bias

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "random_projection",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_scale": self.input_scale,
            "connectivity": self.connectivity,
            "bias_scale": self.bias_scale,
            "seed": self.seed,
        }


class CenterCropProjection(Projection):
    """
    Fixed cropping projection that selects the central part of the input feature vector.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(input_dim, output_dim)

        if self._output_dim > self.input_dim:
            raise ValueError(
                f"CenterCropProjection: output_dim ({self._output_dim}) cannot be larger "
                f"than input_dim ({self.input_dim})."
            )

        self.start_idx = (self.input_dim - self._output_dim) // 2
        self.end_idx = self.start_idx + self._output_dim

    def _project(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # inputs shape: (Batch, Time, Features)
        if inputs.ndim != 3:
             raise ValueError(f"CenterCropProjection requires 3D input (Batch, Time, Features), got ndim={inputs.ndim}")
        return inputs[..., self.start_idx : self.end_idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "center_crop",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "crop_range": (self.start_idx, self.end_idx),
        }


class ResizeProjection(Projection):
    """
    Projection that resizes (interpolates) the feature dimension to the target size.
    Uses JAX's image resizing capabilities (bilinear interpolation by default).
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(input_dim, output_dim)
        if self._output_dim > self.input_dim:
            raise ValueError(
                f"ResizeProjection: output_dim ({self._output_dim}) cannot be larger "
                f"than input_dim ({self.input_dim})."
            )

    def _project(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # inputs shape: (Batch, Time, Features) or (Batch, Features)
        # We assume the last dimension is Features.
        # jax.image.resize expects shape to resize to.
        
        input_shape = inputs.shape
        # Target shape: preserve all but last dim, replace last with output_dim
        target_shape = input_shape[:-1] + (self._output_dim,)
        
        # 'linear' (bilinear) is standard for resizing. 
        # For 1D signal (Features), linear interpolation is equivalent to resizing along 1 axis.
        return jax.image.resize(inputs, target_shape, method='linear', antialias=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "resize_projection",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }


# --- 3. The Factory Logic (Dependency Injection Helper) ---

@singledispatch
def create_projection(config: Any, input_dim: int) -> Projection:
    """
    Factory function to create a Projection instance based on the config type.
    Raises TypeError if the config type is not registered.
    """
    raise TypeError(f"Unknown projection config type: {type(config)}")

def register_projections(
    CenterCropConfigClass: Type, 
    RandomProjectionConfigClass: Type,
    ResizeProjectionConfigClass: Type
):
    """
    Call this function once to register the handlers.
    """

    @create_projection.register(CenterCropConfigClass)
    def _(config, input_dim: int) -> CenterCropProjection:
        return CenterCropProjection(
            input_dim=int(input_dim),
            output_dim=int(config.n_units),
        )

    @create_projection.register(RandomProjectionConfigClass)
    def _(config, input_dim: int) -> RandomProjection:
        return RandomProjection(
            input_dim=int(input_dim),
            output_dim=int(config.n_units),
            input_scale=float(config.input_scale),
            input_connectivity=float(config.input_connectivity),
            seed=int(config.seed),
            bias_scale=float(config.bias_scale),
        )

    @create_projection.register(ResizeProjectionConfigClass)
    def _(config, input_dim: int) -> ResizeProjection:
        return ResizeProjection(
            input_dim=int(input_dim),
            output_dim=int(config.n_units),
        )

__all__ = ["Projection", "RandomProjection", "CenterCropProjection", "ResizeProjection", "create_projection", "register_projections"]
