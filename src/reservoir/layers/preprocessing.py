"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/layers/preprocessing.py
STEP2 Preprocessing layers.
Here we define preprocessing which doesn't change any dimensions, such as scaling.
"""
from __future__ import annotations
import abc
from functools import singledispatch

from beartype import beartype
import numpy as np
from typing import TYPE_CHECKING

from reservoir.core.types import NpF64

if TYPE_CHECKING:
    from reservoir.core.types import ConfigDict


# --- 1. Interface Definition ---

@beartype
class Preprocessor(abc.ABC):
    """
    Abstract Base Class for Preprocessing Layers.
    Defines the contract that all preprocessing layers must follow.
    Preprocessors do NOT change dimensions.
    """

    @abc.abstractmethod
    def fit(self, X: NpF64) -> Preprocessor:
        """Fit the preprocessor on training data."""

    @abc.abstractmethod
    def transform(self, X: NpF64) -> NpF64:
        """Apply the transformation."""

    @abc.abstractmethod
    def inverse_transform(self, X: NpF64) -> NpF64:
        """Reverse the transformation."""

    @abc.abstractmethod
    def to_dict(self) -> ConfigDict:
        """Serialize configuration parameters."""

    def fit_transform(self, X: NpF64) -> NpF64:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def __call__(self, X: NpF64) -> NpF64:
        """Alias for transform."""
        return self.transform(X)


# --- 2. Concrete Implementations ---

@beartype
class StandardScaler(Preprocessor):
    """Standard scaler (mean removal and variance scaling)."""

    def __init__(self):
        self.mean_: NpF64 | None = None
        self.scale_: NpF64 | None = None

    def fit(self, X: NpF64) -> StandardScaler:
        X_np = X

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

    def transform(self, X: NpF64) -> NpF64:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"StandardScaler requires numpy.ndarray for in-place optimization, got {type(X)}")
        
        arr = X
        if self.mean_ is not None:
            arr -= self.mean_
        if self.scale_ is not None:
            arr /= self.scale_
        return arr

    def inverse_transform(self, X: NpF64) -> NpF64:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"StandardScaler requires numpy.ndarray for in-place optimization, got {type(X)}")
            
        arr = X
        if self.scale_ is not None:
            arr *= self.scale_
        if self.mean_ is not None:
            arr += self.mean_
        return arr

    def to_dict(self) -> ConfigDict:
        return {
            "type": "standard_scaler",
        }


@beartype
class MinMaxScaler(Preprocessor):
    """
    Min-Max Scaler with feature range scaling.
    Scales data to [feature_min, feature_max].
    Formula: X_std = (X - X.min) / (X.max - X.min)
             X_scaled = X_std * (feature_max - feature_min) + feature_min
    """

    def __init__(self, feature_min: float , feature_max: float):
        self.feature_min = feature_min
        self.feature_max = feature_max
        self.min_: NpF64 | None = None
        self.range_: NpF64 | None = None  # X_max - X_min

    def fit(self, X: NpF64) -> MinMaxScaler:
        X_np = X

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

    def transform(self, X: NpF64) -> NpF64:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"MinMaxScaler requires numpy.ndarray for in-place optimization, got {type(X)}")
            
        arr = X
        # 1. Scale to [0, 1]
        if self.min_ is not None and self.range_ is not None:
            arr -= self.min_
            arr /= self.range_
        
        # 2. Scale to [feature_min, feature_max]
        scale = self.feature_max - self.feature_min
        arr *= scale
        arr += self.feature_min
        
        return arr

    def inverse_transform(self, X: NpF64) -> NpF64:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"MinMaxScaler requires numpy.ndarray for in-place optimization, got {type(X)}")
            
        arr = X
        # 1. Reverse Scale to [0, 1]
        scale = self.feature_max - self.feature_min
        if scale != 0:
            arr -= self.feature_min
            arr /= scale
            
        # 2. Reverse Scale to Original
        if self.min_ is not None and self.range_ is not None:
            arr *= self.range_
            arr += self.min_
            
        return arr

    def to_dict(self) -> ConfigDict:
        return {
            "type": "min_max_scaler",
            "feature_min": self.feature_min,
            "feature_max": self.feature_max,
        }


@beartype
class IdentityPreprocessor(Preprocessor):
    """No-op preprocessor for RAW mode."""

    def fit(self, X: NpF64) -> IdentityPreprocessor:
        _ = X  # Identity: no fitting needed
        return self

    def transform(self, X: NpF64) -> NpF64:
        return X

    def inverse_transform(self, X: NpF64) -> NpF64:
        return X

    def to_dict(self) -> ConfigDict:
        return {"type": "identity"}




@beartype
class AffineScaler(Preprocessor):
    """
    Affine transformation scaler.
    Formula: X_scaled = X * input_scale + shift
    """

    def __init__(self, input_scale: float, shift: float):
        self.input_scale = input_scale
        self.shift = shift

    def fit(self, X: NpF64) -> AffineScaler:
        # AffineScaler is stateless (parameters are provided at init), so fit does nothing.
        return self

    def transform(self, X: NpF64) -> NpF64:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"AffineScaler requires numpy.ndarray for in-place optimization, got {type(X)}")
        
        arr = X
        arr *= self.input_scale
        arr += self.shift
        return arr

    def inverse_transform(self, X: NpF64) -> NpF64:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"AffineScaler requires numpy.ndarray for in-place optimization, got {type(X)}")
            
        # Avoid division by zero
        if self.input_scale == 0:
            return X
            
        arr = X
        arr -= self.shift
        arr /= self.input_scale
        return arr

    def to_dict(self) -> ConfigDict:
        return {
            "type": "affine_scaler",
            "input_scale": self.input_scale,
            "shift": self.shift,
        }


if TYPE_CHECKING:
    from reservoir.models.config import PreprocessingConfig

# --- 3. Factory Logic (Dependency Injection Helper) ---

@singledispatch
def create_preprocessor(config: PreprocessingConfig) -> Preprocessor:
    """
    Factory function to create a Preprocessor instance based on the config type.
    Raises TypeError if the config type is not registered.
    """
    raise TypeError(f"Unknown preprocessor config type: {type(config)}")


def register_preprocessors(
    RawConfigClass: type,
    StandardScalerConfigClass: type,
    MinMaxScalerConfigClass: type | None = None,
    AffineScalerConfigClass: type | None = None,
):
    """
    Register config classes with the factory.
    Call this once at module initialization.
    """

    @create_preprocessor.register(RawConfigClass)
    def _(_config) -> Preprocessor:
        return IdentityPreprocessor()

    @create_preprocessor.register(StandardScalerConfigClass)
    def _(_config) -> Preprocessor:
        return StandardScaler()

    if MinMaxScalerConfigClass is not None:
        @create_preprocessor.register(MinMaxScalerConfigClass)
        def _(config) -> Preprocessor:
            return MinMaxScaler(feature_min=config.feature_min, feature_max=config.feature_max)

    if AffineScalerConfigClass is not None:
        @create_preprocessor.register(AffineScalerConfigClass)
        def _(config) -> Preprocessor:
            return AffineScaler(input_scale=config.input_scale, shift=config.shift)


__all__ = [
    "Preprocessor",
    "StandardScaler",
    "MinMaxScaler",
    "AffineScaler",
    "IdentityPreprocessor",
    "create_preprocessor",
    "register_preprocessors",
]
