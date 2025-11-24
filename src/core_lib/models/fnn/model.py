"""Feed-forward neural network model for pipeline (b)."""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class FNN(nn.Module):
    """Feed-forward network whose depth/width comes from layer_dims."""

    layer_dims: Sequence[int]
    return_hidden: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = jnp.asarray(x, dtype=jnp.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input (batch, features), got shape {x.shape}")
        if not self.layer_dims:
            raise ValueError("layer_dims must include at least one output dimension")

        hidden_output = None
        for idx, feat in enumerate(self.layer_dims):
            x = nn.Dense(features=feat)(x)
            is_last = idx == len(self.layer_dims) - 1
            if not is_last:
                x = nn.relu(x)
                hidden_output = x

        if hidden_output is None:
            hidden_output = x

        if self.return_hidden:
            return x, hidden_output
        return x
