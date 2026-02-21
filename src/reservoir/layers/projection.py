"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/layers/projection.py
STEP3 Projection which changes the dimension of feature vectors.
"""
from __future__ import annotations
import abc
from functools import singledispatch

from beartype import beartype
import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reservoir.core.types import JaxF64, ConfigDict


# --- 1. Interface Definition ---

@beartype
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

    def __call__(self, inputs: JaxF64) -> JaxF64:
        """
        Apply the projection to the inputs.
        template method: validates input shape then calls _forward.
        """
        arr = inputs
        if arr.ndim not in (2, 3):
             raise ValueError(f"{self.__class__.__name__} expects 2D or 3D input, got shape {arr.shape}")
        return self._project(arr)

    @abc.abstractmethod
    def _project(self, inputs: JaxF64) -> JaxF64:
        """Actual projection logic to be implemented by subclasses."""

    @abc.abstractmethod
    def to_dict(self) -> ConfigDict:
        """Serialize configuration parameters."""


# --- 2. Concrete Implementations ---

@beartype
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

    def _project(self, inputs: JaxF64) -> JaxF64:
        # Template ensures inputs is 2D or 3D ndarray
        if inputs.ndim == 3:
            return jnp.einsum("bti,io->bto", inputs, self.W) + self.bias
        # inputs.ndim == 2
        return jnp.dot(inputs, self.W) + self.bias

    def to_dict(self) -> ConfigDict:
        return {
            "type": "random_projection",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_scale": self.input_scale,
            "connectivity": self.connectivity,
            "bias_scale": self.bias_scale,
            "seed": self.seed,
        }


@beartype
class CoherentDriveProjection(Projection):
    """
    Coherent Drive Projection for Quantum Reservoir Computing.
    
    Applies arcsin transformation to convert linear input values to
    quantum rotation angles that achieve amplitude encoding.
    
    The transformation: θ = 2 * arcsin(clip(x, -1, 1))
    When applied with Ry(θ), the state becomes: ψ = sqrt(1-a²)0 + a1
    
    This is non-periodic (unlike angle embedding) and suitable for
    trending time series like Mackey-Glass.
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
        
        # Use RandomProjection-style linear transformation first
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

    def _project(self, inputs: JaxF64) -> JaxF64:
        # Step 1: Linear projection (like RandomProjection)
        if inputs.ndim == 3:
            linear_out = jnp.einsum("bti,io->bto", inputs, self.W) + self.bias
        else:
            linear_out = jnp.dot(inputs, self.W) + self.bias
        
        # Step 2: Coherent Drive transformation (arcsin)
        # Use tanh for soft saturation to [-1, 1] - preserves information at peaks
        # (Unlike clip which causes information loss for |x| > 1)
        saturated = jnp.tanh(linear_out)
        # θ = 2 * arcsin(a) maps amplitude a to rotation angle θ
        theta = 2.0 * jnp.arcsin(saturated)
        
        return theta

    def to_dict(self) -> ConfigDict:
        return {
            "type": "coherent_drive",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_scale": self.input_scale,
            "connectivity": self.connectivity,
            "bias_scale": self.bias_scale,
            "seed": self.seed,
        }


@beartype
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

    def _project(self, inputs: JaxF64) -> JaxF64:
        # inputs shape: (Batch, Time, Features)
        if inputs.ndim != 3:
             raise ValueError(f"CenterCropProjection requires 3D input (Batch, Time, Features), got ndim={inputs.ndim}")
        return inputs[..., self.start_idx : self.end_idx]

    def to_dict(self) -> ConfigDict:
        return {
            "type": "center_crop",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "crop_range": (self.start_idx, self.end_idx),
        }


@beartype
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

    def _project(self, inputs: JaxF64) -> JaxF64:
        # inputs shape: (Batch, Time, Features) or (Batch, Features)
        # We assume the last dimension is Features.
        # jax.image.resize expects shape to resize to.
        
        input_shape = inputs.shape
        # Target shape: preserve all but last dim, replace last with output_dim
        target_shape = input_shape[:-1] + (self._output_dim,)
        
        # 'linear' (bilinear) is standard for resizing. 
        # For 1D signal (Features), linear interpolation is equivalent to resizing along 1 axis.
        return jax.image.resize(inputs, target_shape, method='linear', antialias=True)

    def to_dict(self) -> ConfigDict:
        return {
            "type": "resize_projection",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }


@beartype
class PolynomialProjection(Projection):
    """
    Polynomial feature expansion projection.
    Expands input features by adding polynomial terms up to specified degree.
    Output dimension = input_dim * degree (+ 1 if include_bias).
    """

    def __init__(self, input_dim: int, degree: int, include_bias: bool) -> None:
        # Output dim = input_dim * degree (linear, squared, cubed, etc.) + optional bias
        output_dim = input_dim * degree + (1 if include_bias else 0)
        super().__init__(input_dim, output_dim)
        self.degree = degree
        self.include_bias = include_bias

    def _project(self, inputs: JaxF64) -> JaxF64:
        # inputs shape: (Batch, Time, Features) or (Batch, Features)
        features = [inputs]
        if self.degree > 1:
            for d in range(2, self.degree + 1):
                features.append(jnp.power(inputs, d))

        out = jnp.concatenate(features, axis=-1)

        if self.include_bias:
            shape = out.shape[:-1] + (1,)
            bias = jnp.ones(shape)
            out = jnp.concatenate([out, bias], axis=-1)

        return out

    def to_dict(self) -> ConfigDict:
        return {
            "type": "polynomial_projection",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "degree": self.degree,
            "include_bias": self.include_bias,
        }


@beartype
class PCAProjection(Projection):
    """
    PCA (Principal Component Analysis) projection.
    Reduces dimensionality by projecting onto top n_units principal components.
    Requires fit() to be called with training data before use.
    
    Pre-condition: Input data must be preprocessed with StandardScaler (zero mean, unit variance).
    This class assumes data is already centered and skips mean computation for efficiency.
    
    Args:
        input_dim: Input feature dimension.
        n_units: Number of principal components to keep.
        input_scaler: Optional scalar to multiply output after projection.
    """

    def __init__(self, input_dim: int, n_units: int, input_scaler: float = 1.0) -> None:
        if n_units > input_dim:
            raise ValueError(
                f"PCAProjection: n_units ({n_units}) cannot exceed input_dim ({input_dim})"
            )
        super().__init__(input_dim, n_units)
        self.n_units = n_units
        self.input_scaler = float(input_scaler)
        self._components: JaxF64 | None = None  # (n_units, input_dim)
        self._fitted = False

    def fit(self, X: JaxF64) -> PCAProjection:
        """
        Fit PCA on training data.
        X shape: (Time, Features) or (Batch, Time, Features)
        
        Note: Assumes input is already zero-mean (from StandardScaler).
        """
        # Flatten to 2D if 3D
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])  # (Batch*Time, Features)
        
        # Compute covariance directly: Sigma = X.T @ X / (N-1)
        # (mean is already 0 from StandardScaler, no need for jnp.cov overhead)
        n_samples = X.shape[0]
        cov = jnp.dot(X.T, X) / (n_samples - 1)
        eigenvalues, eigenvectors = jnp.linalg.eigh(cov)
        
        # Sort by eigenvalue (descending) and take top n_units
        idx = jnp.argsort(eigenvalues)[::-1]
        self._components = eigenvectors[:, idx[:self.n_units]].T  # (n_units, input_dim)
        self._fitted = True
        return self

    def _project(self, inputs: JaxF64) -> JaxF64:
        if not self._fitted or self._components is None:
            raise RuntimeError("PCAProjection: fit() must be called before projection")
        
        # Direct projection (no centering needed - data is already zero-mean)
        projected = jnp.dot(inputs, self._components.T)
        return projected * self.input_scaler

    def to_dict(self) -> ConfigDict:
        return {
            "type": "pca",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_scaler": self.input_scaler,
        }

if TYPE_CHECKING:
    from reservoir.models.config import ProjectionConfig

# --- 3. The Factory Logic (Dependency Injection Helper) ---

@singledispatch
def create_projection(config: ProjectionConfig, _input_dim: int) -> Projection:
    """
    Factory function to create a Projection instance based on the config type.
    Raises TypeError if the config type is not registered.
    """
    raise TypeError(f"Unknown projection config type: {type(config)}")

def register_projections(
    CenterCropConfigClass: type, 
    RandomProjectionConfigClass: type,
    ResizeProjectionConfigClass: type,
    PolynomialProjectionConfigClass: type | None = None,
    PCAProjectionConfigClass: type | None = None,
    CoherentDriveConfigClass: type | None = None,
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

    if PolynomialProjectionConfigClass is not None:
        @create_projection.register(PolynomialProjectionConfigClass)
        def _(config, input_dim: int) -> PolynomialProjection:
            return PolynomialProjection(
                input_dim=int(input_dim),
                degree=int(config.degree),
                include_bias=bool(config.include_bias),
            )

    if PCAProjectionConfigClass is not None:
        @create_projection.register(PCAProjectionConfigClass)
        def _(config, input_dim: int) -> PCAProjection:
            return PCAProjection(
                input_dim=int(input_dim),
                n_units=int(config.n_units),
                input_scaler=float(config.input_scaler),
            )


    if CoherentDriveConfigClass is not None:
        @create_projection.register(CoherentDriveConfigClass)
        def _(config, input_dim: int) -> CoherentDriveProjection:
            return CoherentDriveProjection(
                input_dim=int(input_dim),
                output_dim=int(config.n_units),
                input_scale=float(config.input_scale),
                input_connectivity=float(config.input_connectivity),
                seed=int(config.seed),
                bias_scale=float(config.bias_scale),
            )


__all__ = [
    "Projection",
    "RandomProjection",
    "CenterCropProjection",
    "ResizeProjection",
    "PolynomialProjection",
    "PCAProjection",
    "CoherentDriveProjection",
    "create_projection",
    "register_projections"
]
