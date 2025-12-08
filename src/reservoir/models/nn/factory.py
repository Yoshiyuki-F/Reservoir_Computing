"""src/reservoir/models/nn/factory.py"""
from __future__ import annotations
from typing import Any, Dict

from reservoir.training.presets import TrainingConfig
from .rnn import RNNModel
from .fnn import FNNModel
from ..config import FNNConfig


class NNModelFactory:
    """Internal Factory to build pure FNN/RNN models (without distillation logic)."""

    @staticmethod
    def create_fnn(model_cfg: FNNConfig, training_cfg: TrainingConfig, *, input_dim: int, output_dim: int) -> FNNModel:
        if not isinstance(model_cfg, FNNConfig):
            raise TypeError(f"create_fnn expects FNNConfig, got {type(model_cfg)}.")
        hidden_layers = tuple(int(h) for h in model_cfg.hidden_layers)
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive for FNN creation.")
        layer_dims = (int(input_dim), *hidden_layers, int(output_dim))
        cfg_dict = {"layer_dims": layer_dims}
        return FNNModel(cfg_dict, training_cfg)

    @staticmethod
    def create_rnn(model_cfg: Dict[str, Any], training_cfg: TrainingConfig) -> RNNModel:
        required = ("input_dim", "hidden_dim", "output_dim")
        missing = [key for key in required if key not in model_cfg]
        if missing:
            raise ValueError(f"RNN model requires keys {missing} in model config.")
        
        # Normalize config structure for RNNModel
        cfg = {
            "input_dim": model_cfg["input_dim"],
            "hidden_dim": model_cfg["hidden_dim"],
            "output_dim": model_cfg["output_dim"],
            "return_sequences": model_cfg.get("return_sequences", False),
            "return_hidden": False,
        }
        return RNNModel(cfg, training_cfg)
