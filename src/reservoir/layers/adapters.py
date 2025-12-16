"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/layers/adapters.py
Step 4 Adapters between architectural steps.
"""
import jax.numpy as jnp
import numpy as np

class Flatten:
    """
    Structural adapter to flatten inputs between architectural steps.
    Transforms (Batch, Time, Feat) -> (Batch, Time * Feat).
    """
    def fit(self, X):
        return self
    
    def transform(self, X):
        X = jnp.asarray(X)
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        if X.ndim == 2:
            return X.reshape(X.shape[0], -1)
        return X.flatten()

    def __call__(self, X):
        return self.transform(X)


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

    def transform(self, X, flatten_batch: bool = True):
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
            
        return X_embedded

    def __call__(self, X, flatten_batch: bool = True):
        return self.transform(X, flatten_batch=flatten_batch)
