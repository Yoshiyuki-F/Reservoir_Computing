"""/home/yoshi/PycharmProjects/Reservoir/src/core_lib/models/factories.py
Factories to build BaseModel-compatible wrappers for Flax modules."""

from __future__ import annotations

from typing import Any, Dict

from .base import ModelFactory, BaseModel
from .flax_wrapper import FlaxSupervisedModel, FlaxTrainingConfig
from .fnn import FNN, FNNModelConfig
from .rnn import SimpleRNN, SimpleRNNConfig


class FlaxModelFactory(ModelFactory):
    """Create FlaxSupervisedModel instances from config dictionaries."""

    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        model_type = config.get("type")
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})

        if model_type == "fnn":
            fnn_cfg = FNNModelConfig(**model_cfg)
            module = FNN(layer_dims=fnn_cfg.layer_dims, return_hidden=False)
            flax_train_cfg = FlaxTrainingConfig(
                **{**training_cfg, "classification": training_cfg.get("classification", True)}
            )
            return FlaxSupervisedModel(module, flax_train_cfg)

        if model_type == "rnn":
            rnn_cfg = SimpleRNNConfig(**model_cfg)
            module = SimpleRNN(
                hidden_dim=rnn_cfg.hidden_dim,
                output_dim=rnn_cfg.output_dim,
                return_sequences=rnn_cfg.return_sequences,
                return_hidden=False,
            )
            flax_train_cfg = FlaxTrainingConfig(**training_cfg)
            return FlaxSupervisedModel(module, flax_train_cfg)

        raise ValueError(
            "Unsupported model_type for FlaxModelFactory. "
            "Expected 'fnn' or 'rnn', got "
            f"{model_type!r}"
        )
