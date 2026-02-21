"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/layers/adapters.py
Step 4 Adapters — ABC + concrete implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp
from beartype import beartype
from reservoir.core.types import JaxF64, KwargsDict
from reservoir.core.types import to_np_f64
from reservoir.utils.reporting import print_feature_stats


# ==========================================
# Abstract Base
# ==========================================

@beartype
class Adapter(ABC):
    """ABC for Step 4 structural adapters that reshape data before model layers."""

    @abstractmethod
    def transform(self, X: JaxF64, flatten_batch: bool = True, log_label: str | None = None, params: KwargsDict | None = None) -> JaxF64:
        """Transform input features."""

    @abstractmethod
    def align_targets(self, targets: JaxF64, log_label: str | None = None, params: KwargsDict | None = None) -> JaxF64:
        """Align targets to match transformed features (e.g., trim for windowing)."""

    def __call__(self, X: JaxF64, flatten_batch: bool = True, log_label: str | None = None, params: KwargsDict | None = None) -> JaxF64:
        return self.transform(X, flatten_batch=flatten_batch, log_label=log_label, params=params)


# ==========================================
# Concrete Adapters
# ==========================================

@beartype
class Flatten(Adapter):
    """
    Structural adapter to flatten inputs between architectural steps.
    Transforms (Batch, Time, Feat) -> (Batch, Time * Feat).
    """
    def fit(self) -> Flatten:
        return self

    def transform(self, X: JaxF64, flatten_batch: bool = True, log_label: str | None = None, params: KwargsDict | None = None) -> JaxF64:
        if X.ndim == 3:
            result = X.reshape(X.shape[0], -1)
        elif X.ndim == 2:
            result = X.reshape(X.shape[0], -1)
        else:
            result = X.flatten()

        if log_label is not None:
            print_feature_stats(to_np_f64(result), log_label)

        return result

    def align_targets(self, targets: JaxF64, log_label: str | None = None, params: KwargsDict | None = None) -> JaxF64:
        """No alignment needed for Flatten adapter."""
        if log_label is not None:
            print_feature_stats(to_np_f64(targets), log_label)
        return targets


@beartype
class TimeDelayEmbedding(Adapter):
    """
    Sliding window embedding for time series.
    Transforms (Batch, Time, Feat) -> (Batch * (Time - Window + 1), Window * Feat).
    """
    def __init__(self, window_size: int = 10) -> None:
        print("\n=== Step 4: Adapter ===")
        self.window_size = window_size

    def fit(self) -> TimeDelayEmbedding:
        return self

    def transform(self, X: JaxF64, flatten_batch: bool = True, log_label: str | None = None, params: KwargsDict | None = None) -> JaxF64:
        # Support 2D input (T, F) - treat as single batch
        is_2d = X.ndim == 2
        if is_2d:
            X = X[jnp.newaxis, :, :]  # (1, T, F)

        # Shape: (N, T, F)
        W = self.window_size

        # Use slicing to create windows. JAX handles this efficiently.
        windows = []
        for i in range(W):
            start = i
            end_offset = W - 1 - i

            if end_offset > 0:
                sl = X[:, start : -end_offset, :]
            else:
                sl = X[:, start:, :]
            windows.append(sl)

        # Concat along feature axis -> (N, T', W*F)
        X_embedded = jnp.concatenate(windows, axis=-1)

        if flatten_batch or is_2d:
            X_embedded = X_embedded.reshape(-1, X_embedded.shape[-1])

        if log_label is not None:
            print_feature_stats(to_np_f64(X_embedded), log_label)

        return X_embedded

    def align_targets(self, targets: JaxF64, log_label: str | None = None, params: KwargsDict | None = None) -> JaxF64:
        """Align targets by dropping first (window_size-1) timesteps to match windowed X."""
        W = self.window_size
        # Support 2D (T, Out)
        if targets.ndim == 2:
            result = targets[W-1:, :]
            if log_label is not None:
                print_feature_stats(to_np_f64(result), f"{log_label} (Time-Trimmed)")
            return result
        # 3D (N, T, Out) -> (N, T - W + 1, Out) -> (N * T', Out)
        aligned = targets[:, W-1:, :]
        reshaped = aligned.reshape(-1, aligned.shape[-1])
        if log_label is not None:
            print_feature_stats(to_np_f64(reshaped), f"{log_label} (Time-Trimmed)")
        return reshaped

    def __call__(self, X: JaxF64, flatten_batch: bool = True, log_label: str | None = None, params: KwargsDict | None = None) -> JaxF64:
        return self.transform(X, flatten_batch=flatten_batch, log_label=log_label, params=params)


__all__ = ["Adapter", "Flatten", "TimeDelayEmbedding"]
