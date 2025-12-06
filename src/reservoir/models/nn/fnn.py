"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/nn/fnn.py
FNN BaseModel wrapper using BaseFlaxModel."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import flax.linen as nn
import jax.numpy as jnp

from reservoir.models.nn.base import BaseFlaxModel
from reservoir.training.presets import TrainingConfig

class FNNModel(BaseFlaxModel):
    """Wrap FNNModule with BaseModel API."""

    def __init__(self, model_config: Dict[str, Any], training_config: TrainingConfig):
        if "layer_dims" not in model_config:
            raise ValueError("FNNModel requires 'layer_dims' in model_config.")
        self.layer_dims: Sequence[int] = tuple(int(v) for v in model_config["layer_dims"])
        if len(self.layer_dims) < 2:
            raise ValueError("FNNModel.layer_dims must include at least input and output dimensions.")
        super().__init__(model_config, training_config)

    def _flatten_inputs(self, X: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(X)
        if arr.ndim == 3:
            return arr.reshape(arr.shape[0], -1)
        if arr.ndim != 2:
            raise ValueError(f"FNNModel expects 2D or 3D input, got shape {arr.shape}")
        return arr

    def train(self, inputs: jnp.ndarray, targets: Optional[jnp.ndarray] = None, **kwargs: Any) -> Dict[str, Any]:
        flat_inputs = self._flatten_inputs(inputs)
        return super().train(flat_inputs, targets, **kwargs)

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        flat_inputs = self._flatten_inputs(X)
        return super().predict(flat_inputs)

    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        flat_inputs = self._flatten_inputs(X)
        return super().evaluate(flat_inputs, y)

    def _create_model_def(self) -> nn.Module:
        return FNN(layer_dims=self.layer_dims, return_hidden=False)


class FNN(nn.Module):
    """Feed-forward network whose depth/width comes from layer_dims."""

    layer_dims: Sequence[int]
    return_hidden: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = jnp.asarray(x, dtype=jnp.float64)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input (batch, features), got shape {x.shape}")
        if len(self.layer_dims) < 2:
            raise ValueError("layer_dims must include at least input and output dimensions")
        if any(dim <= 0 for dim in self.layer_dims):
            raise ValueError(f"All layer_dims must be positive, got {self.layer_dims}")
        if x.shape[1] != self.layer_dims[0]:
            raise ValueError(
                f"Input feature dimension mismatch: expected {self.layer_dims[0]}, received {x.shape[1]}."
            )

        hidden_output = None
        target_dims = self.layer_dims[1:]  # skip input dimension; flax infers input shape
        for idx, feat in enumerate(target_dims):
            x = nn.Dense(features=feat)(x)
            is_last = idx == len(target_dims) - 1
            if not is_last:
                x = nn.relu(x)
                hidden_output = x

        if hidden_output is None:
            hidden_output = x

        if self.return_hidden:
            return x, hidden_output
        return x

