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

class FeatureScaler(Preprocessor):
    """Standard scaler (mean removal and variance scaling)."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: jnp.ndarray) -> "FeatureScaler":
        X_np = np.asarray(X)

        if X_np.ndim == 3:
            reduce_axis = (0, 1)
        else:
            reduce_axis = 0

        if self.with_mean:
            self.mean_ = np.mean(X_np, axis=reduce_axis)
        if self.with_std:
            self.scale_ = np.std(X_np, axis=reduce_axis)
            if self.scale_ is not None:
                self.scale_[self.scale_ == 0] = 1.0
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
            "type": "feature_scaler",
            "with_mean": self.with_mean,
            "with_std": self.with_std,
        }


class MaxScaler(Preprocessor):
    """Scales data by dividing by the maximum value."""

    def __init__(self):
        self.max_val: Optional[float] = None

    def fit(self, X: jnp.ndarray) -> "MaxScaler":
        self.max_val = float(jnp.max(X))
        return self

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        if self.max_val is not None and self.max_val != 0:
            return jnp.asarray(X) / self.max_val
        return jnp.asarray(X)

    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        if self.max_val is not None:
            return jnp.asarray(X) * self.max_val
        return jnp.asarray(X)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "max_scaler", "max_val": self.max_val}


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
    MaxScalerConfigClass: Type,
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
        return FeatureScaler()

    @create_preprocessor.register(MaxScalerConfigClass)
    def _(config) -> Preprocessor:
        return MaxScaler()


__all__ = [
    "Preprocessor",
    "FeatureScaler",
    "MaxScaler",
    "IdentityPreprocessor",
    "create_preprocessor",
    "register_preprocessors",
]
