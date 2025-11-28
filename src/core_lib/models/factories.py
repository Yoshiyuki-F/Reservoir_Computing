"""/home/yoshi/PycharmProjects/Reservoir/src/core_lib/models/factories.py
Factories to build BaseModel-compatible wrappers for Flax modules."""

from __future__ import annotations

from typing import Any, Dict

from .base import ModelFactory, BaseModel
from .nn.fnn import FNNModel
from .nn.rnn import RNNModel
from .nn.config import FNNModelConfig, SimpleRNNConfig


class FlaxModelFactory(ModelFactory):
    """Create BaseFlaxModel instances (FNN/RNN) from config dictionaries."""

    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        model_type = config.get("type")
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})

        if model_type == "fnn":
            fnn_cfg = FNNModelConfig(**model_cfg)
            cfg = {
                "layer_dims": fnn_cfg.layer_dims,
                **training_cfg,
            }
            return FNNModel(cfg)

        if model_type == "rnn":
            rnn_cfg = SimpleRNNConfig(**model_cfg)
            cfg = {
                "input_dim": rnn_cfg.input_dim,
                "hidden_dim": rnn_cfg.hidden_dim,
                "output_dim": rnn_cfg.output_dim,
                "return_sequences": rnn_cfg.return_sequences,
                "return_hidden": False,
                **training_cfg,
            }
            return RNNModel(cfg)

        raise ValueError(
            "Unsupported model_type for FlaxModelFactory. "
            "Expected 'fnn' or 'rnn', got "
            f"{model_type!r}"
        )
