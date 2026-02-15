"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/layers/preprocessing.py
STEP2 Preprocessing layers.
Here we define preprocessing which doesn't change any dimensions, such as scaling.
"""
from __future__ import annotations
import abc
from functools import singledispatch
from typing import Any, Dict, Optional, Type

import jax.numpy as jnp
import numpy as np


# --- 1. Interface Definition ---

class Preprocessor(abc.ABC):
    """
    Abstract Base Class for Preprocessing Layers.
    Defines the contract that all preprocessing layers must follow.
    Preprocessors do NOT change dimensions.
    """

    @abc.abstractmethod
    def fit(self, X: jnp.ndarray) -> "Preprocessor":
        """Fit the preprocessor on training data."""
        pass

    @abc.abstractmethod
    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """Apply the transformation."""
        pass

    @abc.abstractmethod
    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """Reverse the transformation."""
        pass

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration parameters."""
        pass

    def fit_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """Alias for transform."""
        return self.transform(X)


# --- 2. Concrete Implementations ---

class StandardScaler(Preprocessor):
    """Standard scaler (mean removal and variance scaling)."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: jnp.ndarray) -> "StandardScaler":
        X_np = np.asarray(X)

        if X_np.ndim == 3:
            reduce_axis = (0, 1)
        else:
            reduce_axis = 0

        self.mean_ = np.mean(X_np, axis=reduce_axis)
        self.scale_ = np.std(X_np, axis=reduce_axis)
        
        # Avoid division by zero and extreme scaling for near-constant features
        if self.scale_ is not None:
            self.scale_[self.scale_ < 1e-6] = 1.0
            
        return self

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(X)
        if self.mean_ is not None:
            arr = arr - self.mean_
        if self.scale_ is not None:
            arr = arr / self.scale_
        return arr

    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(X)
        if self.scale_ is not None:
            arr = arr * self.scale_
        if self.mean_ is not None:
            arr = arr + self.mean_
        return arr

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "standard_scaler",
        }


class CustomRangeScaler(Preprocessor):
    """
    Scales data by dividing by the maximum value and multiplying by a scalar.
    Formula: X_scaled = (X / max(X)) * input_scale
    
    When centering=True:
    Formula: X_scaled = ((X - mean(X)) / max(|X - mean(X)|)) * input_scale
    """

    def __init__(self, input_scale: float, centering: bool):
        self.input_scale = input_scale
        self.centering = centering
        self.max_val: Optional[float] = None
        self.mean_: Optional[np.ndarray] = None

    def fit(self, X: jnp.ndarray) -> "CustomRangeScaler":
        X_np = np.asarray(X)
        
        if self.centering:
            if X_np.ndim == 3:
                reduce_axis = (0, 1)
            else:
                reduce_axis = 0
            self.mean_ = np.mean(X_np, axis=reduce_axis)
            # Calculate max of centered data (absolute max)
            self.max_val = float(np.max(np.abs(X_np - self.mean_)))
        else:
            self.max_val = float(np.max(X_np))
            
        return self

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(X)
        
        # 1. Center (if enabled)
        if self.centering and self.mean_ is not None:
            arr = arr - self.mean_

        # 2. Scale to Unit (X / max_val)
        if self.max_val is not None and self.max_val != 0:
            arr = arr / self.max_val
            
        # 3. Apply Custom Range Scale
        if self.input_scale != 1.0:
            arr = arr * self.input_scale
            
        return arr

    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(X)
        
        # 1. Remove Custom Range Scale
        if self.input_scale != 1.0 and self.input_scale != 0:
            arr = arr / self.input_scale

        # 2. Un-scale (multiply by max)
        if self.max_val is not None:
            arr = arr * self.max_val

        # 3. Un-center
        if self.centering and self.mean_ is not None:
            arr = arr + self.mean_
            
        return arr

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "custom_range_scaler",
            "input_scale": self.input_scale,
            "max_val": self.max_val,
            "centering": self.centering
        }


class MinMaxScaler(Preprocessor):
    """
    Min-Max Scaler with input scaling (Murauer et al., 2025).
    Formula: s_k = (P(t_k) - P_min) / (P_max - P_min)
    Maps data to [0, input_scale] range.
    """

    def __init__(self, input_scale):
        self.input_scale = input_scale #a_in in the paper
        self.min_: Optional[np.ndarray] = None
        self.range_: Optional[np.ndarray] = None  # P_max - P_min

    def fit(self, X: jnp.ndarray) -> "MinMaxScaler":
        X_np = np.asarray(X)

        if X_np.ndim == 3:
            reduce_axis = (0, 1)
        else:
            reduce_axis = 0

        self.min_ = np.min(X_np, axis=reduce_axis)
        max_ = np.max(X_np, axis=reduce_axis)
        self.range_ = max_ - self.min_

        # Avoid division by zero for constant features
        if self.range_ is not None:
            self.range_[self.range_ < 1e-12] = 1.0

        return self

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(X)
        if self.min_ is not None and self.range_ is not None:
            arr = (arr - self.min_) / self.range_
        arr = arr * self.input_scale
        return arr

    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(X)
        if self.input_scale != 0:
            arr = arr / self.input_scale
        if self.min_ is not None and self.range_ is not None:
            arr = arr * self.range_ + self.min_
        return arr

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "min_max_scaler",
            "input_scale": self.input_scale,
        }


class IdentityPreprocessor(Preprocessor):
    """No-op preprocessor for RAW mode."""

    def fit(self, X: jnp.ndarray) -> "IdentityPreprocessor":
        return self

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(X)

    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(X)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "identity"}




class AffineScaler(Preprocessor):
    """
    Affine transformation scaler.
    Formula: X_scaled = X * input_scale + shift
    """

    def __init__(self, input_scale: float, shift: float):
        self.input_scale = input_scale
        self.shift = shift

    def fit(self, X: jnp.ndarray) -> "AffineScaler":
        # AffineScaler is stateless (parameters are provided at init), so fit does nothing.
        return self

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        return X * self.input_scale + self.shift

    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        # Avoid division by zero
        if self.input_scale == 0:
            return X
        return (X - self.shift) / self.input_scale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "affine_scaler",
            "input_scale": self.input_scale,
            "shift": self.shift,
        }


# --- 3. Factory Logic (Dependency Injection Helper) ---

@singledispatch
def create_preprocessor(config: Any) -> Preprocessor:
    """
    Factory function to create a Preprocessor instance based on the config type.
    Raises TypeError if the config type is not registered.
    """
    raise TypeError(f"Unknown preprocessor config type: {type(config)}")


def register_preprocessors(
    RawConfigClass: Type,
    StandardScalerConfigClass: Type,
    CustomRangeScalerConfigClass: Type,
    MinMaxScalerConfigClass: Type = None,
    AffineScalerConfigClass: Type = None,
):
    """
    Register config classes with the factory.
    Call this once at module initialization.
    """

    @create_preprocessor.register(RawConfigClass)
    def _(config) -> Preprocessor:
        return IdentityPreprocessor()

    @create_preprocessor.register(StandardScalerConfigClass)
    def _(config) -> Preprocessor:
        return StandardScaler()

    @create_preprocessor.register(CustomRangeScalerConfigClass)
    def _(config) -> Preprocessor:
        return CustomRangeScaler(input_scale=config.input_scale, centering=config.centering)

    if MinMaxScalerConfigClass is not None:
        @create_preprocessor.register(MinMaxScalerConfigClass)
        def _(config) -> Preprocessor:
            return MinMaxScaler(input_scale=config.input_scale)

    if AffineScalerConfigClass is not None:
        @create_preprocessor.register(AffineScalerConfigClass)
        def _(config) -> Preprocessor:
            return AffineScaler(input_scale=config.input_scale, shift=config.shift)


__all__ = [
    "Preprocessor",
    "StandardScaler",
    "CustomRangeScaler",
    "MinMaxScaler",
    "AffineScaler",
    "IdentityPreprocessor",
    "create_preprocessor",
    "register_preprocessors",
]
