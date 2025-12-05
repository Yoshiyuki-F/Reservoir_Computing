"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/factories.py
Factories to build Flax BaseModel-compatible wrappers from validated configs."""

from __future__ import annotations

from typing import Any, Dict

from reservoir.training.presets import TrainingConfig
from .nn.base import BaseModel
from .nn.rnn import RNNModel
from .nn.fnn import FNNModel
from .distillation import DistillationModel
from .presets import DistillationConfig, ReservoirConfig


class FlaxModelFactory:
    """Create BaseModel-compatible instances (FNN/RNN/Distillation) from config dictionaries."""

    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        model_type = config.get("type")
        model_cfg = dict(config.get("model", {}) or {})
        training_cfg = config.get("training")
        if not isinstance(training_cfg, TrainingConfig):
            raise TypeError("FlaxModelFactory expects 'training' to be a TrainingConfig instance.")

        if model_type == "fnn":
            return FlaxModelFactory._create_fnn(model_cfg, training_cfg, config)

        if model_type == "rnn":
            return FlaxModelFactory._create_rnn(model_cfg, training_cfg)

        raise ValueError(
            "Unsupported model_type for FlaxModelFactory. "
            "Expected 'fnn' or 'rnn', got "
            f"{model_type!r}"
        )

    @staticmethod
    def _create_fnn(model_cfg: Dict[str, Any], training_cfg: TrainingConfig, full_config: Dict[str, Any]) -> BaseModel:
        if "layer_dims" not in model_cfg:
            raise ValueError("FNN model requires 'layer_dims' list in model config.")
        fnn_cfg = model_cfg

        reservoir_cfg_dict: Dict[str, Any] = full_config.get("reservoir") or full_config.get("reservoir_params") or {}
        if reservoir_cfg_dict:
            hidden_dim_override = full_config.get("hidden_dim")
            if hidden_dim_override is not None:
                reservoir_cfg_dict = dict(reservoir_cfg_dict)
                reservoir_cfg_dict.setdefault("n_units", hidden_dim_override)

            teacher_cfg = ReservoirConfig(**reservoir_cfg_dict)
            teacher_cfg.validate(context="distillation.teacher")
            distill_cfg = DistillationConfig(teacher=teacher_cfg, student_hidden_layers=tuple(int(v) for v in fnn_cfg["layer_dims"][1:-1] or ()))

            input_dim = FlaxModelFactory._validate_input_dim(full_config, fnn_cfg)
            print(">> Factory: FNN model operating in Distillation mode (Teacher=Reservoir, Student=FNN).")
            return DistillationModel(distill_cfg, training_cfg, int(input_dim))

        return FNNModel(fnn_cfg, training_cfg)

    @staticmethod
    def _create_rnn(model_cfg: Dict[str, Any], training_cfg: TrainingConfig) -> BaseModel:
        required = ("input_dim", "hidden_dim", "output_dim")
        missing = [key for key in required if key not in model_cfg]
        if missing:
            raise ValueError(f"RNN model requires keys {missing} in model config.")
        cfg = {
            "input_dim": model_cfg["input_dim"],
            "hidden_dim": model_cfg["hidden_dim"],
            "output_dim": model_cfg["output_dim"],
            "return_sequences": model_cfg.get("return_sequences", False),
            "return_hidden": False,
        }
        return RNNModel(cfg, training_cfg)

    @staticmethod
    def _validate_input_dim(full_config: Dict[str, Any], fnn_cfg: Dict[str, Any]) -> int:
        layer_dims = tuple(int(v) for v in fnn_cfg["layer_dims"])
        config_input_dim = full_config.get("input_dim")
        inferred_input_dim = layer_dims[0] if layer_dims else None

        if config_input_dim is not None and inferred_input_dim is not None:
            if int(config_input_dim) != int(inferred_input_dim):
                raise ValueError(
                    "DistillationModel input dimension mismatch: "
                    f"config['input_dim']={config_input_dim} vs FNN layer_dims[0]={inferred_input_dim}."
                )

        input_dim = config_input_dim or inferred_input_dim
        if input_dim is None:
            raise ValueError(
                "Distillation model requires 'input_dim' to be specified via config['input_dim'] "
                "or inferred from the first FNN layer dimension."
            )
        return int(input_dim)
