"""Feed-forward neural network model for pipeline (b)."""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class FNN(nn.Module):
    """Simple feed-forward network used as a feature extractor."""

    features: Sequence[int]
    return_hidden: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = jnp.asarray(x, dtype=jnp.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input (batch, features), got shape {x.shape}")
        for feat in self.features[:-1]:
            x = nn.Dense(features=feat)(x)
            x = nn.relu(x)
        hidden = x
        x = nn.Dense(features=self.features[-1])(x)
        if self.return_hidden:
            return x, hidden
        return x

