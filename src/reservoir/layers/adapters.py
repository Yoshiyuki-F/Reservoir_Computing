import jax.numpy as jnp

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
            return X.reshape(-1)
        return X.flatten()

    def __call__(self, X):
        return self.transform(X)
