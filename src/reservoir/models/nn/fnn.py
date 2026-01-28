"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/nn/fnn.py
FNN BaseModel wrapper using BaseFlaxModel.
Adapter is selected based on FNNConfig: Flatten (default) or TimeDelayEmbedding (if window_size is set).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Optional

import flax.linen as nn
import jax.numpy as jnp

from reservoir.models.config import FNNConfig
from reservoir.models.nn.base import BaseFlaxModel
from reservoir.training.presets import TrainingConfig
from reservoir.layers.adapters import Flatten, TimeDelayEmbedding


class FNNModel(BaseFlaxModel):
    """
    FNN with configurable adapter (Step 4).
    
    If config.window_size is None: uses Flatten adapter (for classification)
    If config.window_size is set: uses TimeDelayEmbedding adapter (for time series regression)
    """

    def __init__(self, model_config: FNNConfig, training_config: TrainingConfig, input_dim: int, output_dim: int, classification: bool = False):
        if not isinstance(model_config, FNNConfig):
            raise TypeError(f"FNNModel expects FNNConfig, got {type(model_config)}.")
        if int(input_dim) <= 0 or int(output_dim) <= 0:
            raise ValueError("input_dim and output_dim must be positive for FNNModel.")

        # Store window configuration
        self.window_size = model_config.window_size
        
        # Select adapter based on config
        if self.window_size is not None:
            # TimeDelayEmbedding: input_dim becomes window_size * original_features
            self.adapter = TimeDelayEmbedding(window_size=self.window_size)
            effective_input_dim = self.window_size * int(input_dim)
        else:
            # Flatten: standard behavior
            self.adapter = Flatten()
            effective_input_dim = int(input_dim)

        hidden_layers = tuple(int(h) for h in (model_config.hidden_layers or ()))
        hidden_layers = tuple(h for h in hidden_layers if h > 0)

        self.layer_dims: Sequence[int] = (effective_input_dim, *hidden_layers, int(output_dim))

        super().__init__({"layer_dims": self.layer_dims}, training_config, classification=classification)

    def train(self, inputs: jnp.ndarray, targets: Optional[jnp.ndarray] = None, **kwargs: Any) -> Dict[str, Any]:
        """Train with adapter-transformed inputs (and aligned targets if windowed)."""
        # Log Step 4 (Adapter) only during training
        log_label = f"4:TimeDelayEmbedding(k={self.window_size}):train" if self.window_size else None
        adapted_inputs = self.adapter(inputs, log_label=log_label)
        aligned_targets = self.adapter.align_targets(targets) if targets is not None else None

        return super().train(adapted_inputs, aligned_targets, **kwargs)

    def predict(self, X: jnp.ndarray, **kwargs: Any) -> jnp.ndarray:
        """Predict with adapter-transformed inputs."""
        adapted_inputs = self.adapter(X)
        return super().predict(adapted_inputs)

    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        """Evaluate with adapter-transformed inputs (and aligned targets if windowed)."""
        adapted_inputs = self.adapter(X)
        aligned_targets = self.adapter.align_targets(y)

        return super().evaluate(adapted_inputs, aligned_targets)

    def __call__(self, X: jnp.ndarray, **kwargs: Any) -> jnp.ndarray:
        """Make model callable for batched_compute compatibility."""
        return self.predict(X)

    def _create_model_def(self) -> nn.Module:
        return FNN(layer_dims=self.layer_dims, return_hidden=False)


class FNN(nn.Module):
    """Feed-forward network whose depth/width comes from layer_dims."""

    layer_dims: Sequence[int]
    return_hidden: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = jnp.asarray(x, dtype=jnp.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input (batch, features), got shape {x.shape}")
        if len(self.layer_dims) < 2:
            raise ValueError("layer_dims must include at least input and output dimensions")

        hidden_output = None
        # 入力次元(layer_dims[0])はスキップし、隠れ層以降の次元のみを使用する
        target_dims = self.layer_dims[1:]

        for idx, feat in enumerate(target_dims):
            # nn.Denseは入力次元を自動推論するため、features（出力次元）だけ指定すればOK
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