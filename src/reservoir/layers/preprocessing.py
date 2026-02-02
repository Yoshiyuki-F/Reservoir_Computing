"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/layers/preprocessing.py
Step 2 Preprocessing layers.
Here we define preprocessing which doesnt change any dimensions, such as scaling.
"""
import numpy as np
import jax.numpy as jnp
from typing import Optional, Union, List, Any
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

        if X.ndim == 3:
            reduce_axis = (0, 1)
        else:
            reduce_axis = 0

        if self.with_mean:
            self.mean_ = np.mean(X, axis=reduce_axis)
        if self.with_std:
            self.scale_ = np.std(X, axis=reduce_axis)
            if self.scale_ is not None:
                self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        X = jnp.array(X)
        if self.mean_ is not None:
            X = X - self.mean_
        if self.scale_ is not None:
            X = X / self.scale_
        return jnp.asarray(X)

    def inverse_transform(self, X: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        X = jnp.array(X)
        if self.scale_ is not None:
            X = X * self.scale_
        if self.mean_ is not None:
            X = X + self.mean_
        return jnp.asarray(X)

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

    @staticmethod
    def inverse_transform(X):
        return X

    def __call__(self, X):
        return self.transform(X)


class MaxScaler:
    def __init__(self):
        self.max_val = None

    def fit(self, X):
        # Trainデータの最大値を「記憶」する
        self.max_val = jnp.max(X)
        return self

    def transform(self, X):
        # 記憶しておいたTrainの最大値で割る
        # (Testデータにこれより大きい値が来たら 1.0 を超えるが、それは許容する)
        if self.max_val is not None:
            return X / self.max_val
        return X

    def inverse_transform(self, X):
        if self.max_val is not None:
            return X * self.max_val
        return X


def create_preprocessor(p_type: Preprocessing, poly_degree:int) -> tuple[List[object], list[str]]:
    """
    Build a list of preprocessing transformers for Step 2 based on Enum only.
    RAW: no-op
    STANDARD_SCALER: FeatureScaler
    DESIGN_MATRIX: FeatureScaler + DesignMatrix
    """
    layers: List[Any] = []
    preprocess_labels: list[str] = []
    if p_type == Preprocessing.STANDARD_SCALER:
        layers.append(FeatureScaler())
        preprocess_labels.append("scaler")
    elif p_type == Preprocessing.DESIGN_MATRIX:
        #多項式拡張はスケーリング必須のため
        layers.append(FeatureScaler())
        layers.append(DesignMatrix(degree=poly_degree))
        preprocess_labels.extend(["scaler", f"poly{poly_degree}"])
    elif p_type == Preprocessing.MAX_SCALER:
        layers.append(MaxScaler())
        preprocess_labels.append("max_scaler")
    # RAW -> no layers
    return layers, preprocess_labels



def apply_layers(layers: List[Any], data: np.ndarray, *, fit: bool = False) -> jnp.ndarray:
    """Sequentially apply preprocessing layers."""
    arr = jnp.asarray(data)  # Convert to JAX at start
    for layer in layers:
        if fit and hasattr(layer, "fit_transform"):
            arr = layer.fit_transform(arr)
            fit = False
        elif fit and hasattr(layer, "fit") and hasattr(layer, "transform"):
            layer.fit(arr)
            arr = layer.transform(arr)
            fit = False
        elif hasattr(layer, "transform"):
            arr = layer.transform(arr)
        else:
            arr = layer(arr)
    return arr  # Keep as JAX array

__all__ = ["FeatureScaler", "DesignMatrix", "create_preprocessor", "apply_layers"]
