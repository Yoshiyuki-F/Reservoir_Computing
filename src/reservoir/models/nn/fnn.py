"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/nn/fnn.py
FNN BaseModel wrapper using BaseFlaxModel and Flatten Adapter.
Input dimension check removed to allow Flax lazy initialization.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Optional

import flax.linen as nn
import jax.numpy as jnp

from reservoir.models.config import FNNConfig
from reservoir.models.nn.base import BaseFlaxModel
from reservoir.training.presets import TrainingConfig
from reservoir.layers.adapters import Flatten

class FNNModel(BaseFlaxModel):
    """Wrap FNNModule with BaseModel API using Flatten Adapter."""

    def __init__(self, model_config: FNNConfig, training_config: TrainingConfig, input_dim: int, output_dim: int, classification: bool = False):
        if not isinstance(model_config, FNNConfig):
            raise TypeError(f"FNNModel expects FNNConfig, got {type(model_config)}.")
        # input_dimの正当性チェックは残すが、実際のデータ形状との整合性はFNNクラスに任せる
        if int(input_dim) <= 0 or int(output_dim) <= 0:
            raise ValueError("input_dim and output_dim must be positive for FNNModel.")

        hidden_layers = tuple(int(h) for h in (model_config.hidden_layers or ()))
        hidden_layers = tuple(h for h in hidden_layers if h > 0)

        # input_dimはConfigとして保存するが、実際の初期化時には無視される（Flaxの推論に任せる）
        self.layer_dims: Sequence[int] = (int(input_dim), *hidden_layers, int(output_dim))

        super().__init__({"layer_dims": self.layer_dims}, training_config, classification=classification)
        self.adapter = Flatten()

    def train(self, inputs: jnp.ndarray, targets: Optional[jnp.ndarray] = None, **kwargs: Any) -> Dict[str, Any]:
        flat_inputs = self.adapter(inputs)
        return super().train(flat_inputs, targets, **kwargs)

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        flat_inputs = self.adapter(X)
        return super().predict(flat_inputs)

    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        flat_inputs = self.adapter(X)
        return super().evaluate(flat_inputs, y)

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