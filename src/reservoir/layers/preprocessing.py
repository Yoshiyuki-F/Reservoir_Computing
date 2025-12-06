"""Step 2 Preprocessing layers."""
import numpy as np
import jax.numpy as jnp
from typing import Optional, Union, List
from reservoir.core.identifiers import Preprocessing


class FeatureScaler:
    """Standard scaler (mean removal and variance scaling)."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, jnp.ndarray]) -> "FeatureScaler":
        X = np.asarray(X)
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        if self.with_std:
            self.scale_ = np.std(X, axis=0)
            self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        X = jnp.array(X)
        if self.mean_ is not None:
            X = X - self.mean_
        if self.scale_ is not None:
            X = X / self.scale_
        return X

    def fit_transform(self, X: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        return self.fit(X).transform(X)

    def __call__(self, X):
        return self.transform(X)


class DesignMatrix:
    """Polynomial feature expansion."""

    def __init__(self, degree: int = 1, include_bias: bool = True):
        self.degree = degree
        self.include_bias = include_bias

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        # X shape: (time, features) or (batch, time, features)
        features = [X]
        if self.degree > 1:
            for d in range(2, self.degree + 1):
                features.append(jnp.power(X, d))

        out = jnp.concatenate(features, axis=-1)

        if self.include_bias:
            shape = out.shape[:-1] + (1,)
            bias = jnp.ones(shape)
            out = jnp.concatenate([out, bias], axis=-1)

        return out

    def __call__(self, X):
        return self.transform(X)


def create_preprocessor(p_type: Preprocessing, **kwargs) -> List[object]:
    """
    Build a list of preprocessing transformers for Step 2 based on Enum only.
    RAW: no-op
    STANDARD_SCALER: FeatureScaler
    DESIGN_MATRIX: FeatureScaler + DesignMatrix
    """
    layers: List[object] = []
    if p_type == Preprocessing.STANDARD_SCALER:
        layers.append(FeatureScaler())
    elif p_type == Preprocessing.DESIGN_MATRIX:
        #多項式拡張はスケーリング必須のため
        layers.append(FeatureScaler())
        degree = int(kwargs.get("poly_degree", 2))
        layers.append(DesignMatrix(degree=degree))
    # RAW -> no layers
    return layers


__all__ = ["FeatureScaler", "DesignMatrix", "create_preprocessor"]
