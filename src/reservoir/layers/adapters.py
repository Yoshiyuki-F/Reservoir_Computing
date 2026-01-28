"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/layers/adapters.py
Step 4 Adapters before the actual model layers
"""
import jax.numpy as jnp
from typing import Optional
from reservoir.utils.reporting import print_feature_stats


class Flatten:
    """
    Structural adapter to flatten inputs between architectural steps.
    Transforms (Batch, Time, Feat) -> (Batch, Time * Feat).
    """
    def fit(self):
        return self
    
    @staticmethod
    def transform(X, log_label: Optional[str] = None):
        X = jnp.asarray(X)
        if X.ndim == 3:
            result = X.reshape(X.shape[0], -1)
        elif X.ndim == 2:
            result = X.reshape(X.shape[0], -1)
        else:
            result = X.flatten()
        
        if log_label is not None:
            from reservoir.utils.reporting import print_feature_stats
            print_feature_stats(result, log_label)
        
        return result

    @staticmethod
    def align_targets(targets):
        """No alignment needed for Flatten adapter."""
        return targets

    def __call__(self, X, log_label: Optional[str] = None):
        return self.transform(X, log_label=log_label)


class TimeDelayEmbedding:
    """
    Sliding window embedding for time series.
    Transforms (Batch, Time, Feat) -> (Batch * (Time - Window + 1), Window * Feat).
    Flattening the batch and time dimensions allows standard FNNs to process the windows as independent samples.
    """
    def __init__(self, window_size: int = 10):
        print(f"\n=== Step 4: Adapter ===")
        self.window_size = window_size

    def fit(self):
        return self

    def transform(self, X, flatten_batch: bool = True, log_label: Optional[str] = None):
        X = jnp.asarray(X)

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
            
            # Slice: X[:, start : end]
            # If end_offset > 0, we slice up to -end_offset
            # If end_offset == 0, we slice to the end
            if end_offset > 0:
                sl = X[:, start : -end_offset, :]
            else:
                sl = X[:, start:, :]
            windows.append(sl)
        
        # Concat along feature axis -> (N, T', W*F)
        X_embedded = jnp.concatenate(windows, axis=-1)

        if flatten_batch or is_2d:
            # (N * T', W*F) or (T', W*F) for 2D input
            X_embedded = X_embedded.reshape(-1, X_embedded.shape[-1])
        
        # Log feature stats only when log_label is provided
        if log_label is not None:
            print_feature_stats(X_embedded, log_label)

        return X_embedded

    def align_targets(self, targets, log_label: Optional[str] = None):
        """Align targets by dropping first (window_size-1) timesteps to match windowed X."""
        targets = jnp.asarray(targets)
        W = self.window_size
        # Support 2D (T, Out)
        if targets.ndim == 2:
            result = targets[W-1:, :]
            if log_label is not None:
                print_feature_stats(result, log_label)
            return result
        # 3D (N, T, Out) -> (N, T - W + 1, Out) -> (N * T', Out)
        aligned = targets[:, W-1:, :]
        reshaped = aligned.reshape(-1, aligned.shape[-1])
        if log_label is not None:
            print_feature_stats(reshaped, log_label)
        return reshaped

    def __call__(self, X, flatten_batch: bool = True, log_label: Optional[str] = None):
        return self.transform(X, flatten_batch=flatten_batch, log_label=log_label)

