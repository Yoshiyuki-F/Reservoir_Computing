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
