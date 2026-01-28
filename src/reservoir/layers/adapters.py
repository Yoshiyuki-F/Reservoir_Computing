"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/layers/adapters.py
Step 4 Adapters before the actual model layers
"""
import jax.numpy as jnp
from typing import Optional

class Flatten:
    """
    Structural adapter to flatten inputs between architectural steps.
    Transforms (Batch, Time, Feat) -> (Batch, Time * Feat).
    """
    def fit(self):
        return self
    
    def transform(self, X, log_label: Optional[str] = None):
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

    def __call__(self, X, log_label: Optional[str] = None):
        return self.transform(X, log_label=log_label)


class TimeDelayEmbedding:
    """
    Sliding window embedding for time series.
    Transforms (Batch, Time, Feat) -> (Batch * (Time - Window + 1), Window * Feat).
    Flattening the batch and time dimensions allows standard FNNs to process the windows as independent samples.
    """
    def __init__(self, window_size: int = 10):
        self.window_size = window_size

    def fit(self, X):
        return self

    def transform(self, X, flatten_batch: bool = True, log_label: Optional[str] = None):
        X = jnp.asarray(X)
        if X.ndim != 3:
             # Assuming shape is static or JAX handles error
             pass 
            
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

        if flatten_batch:
            # (N * T', W*F)
            X_embedded = X_embedded.reshape(-1, X_embedded.shape[-1])
        
        if log_label is not None:
            from reservoir.utils.reporting import print_feature_stats
            print_feature_stats(X_embedded, log_label)
            
        return X_embedded

    def __call__(self, X, flatten_batch: bool = True, log_label: Optional[str] = None):
        return self.transform(X, flatten_batch=flatten_batch, log_label=log_label)

