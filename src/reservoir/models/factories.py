"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/factories.py
Factories to build BaseModel-compatible wrappers for Flax modules."""

from __future__ import annotations

from typing import Any, Dict

from .nn.base import BaseModel
from .nn.rnn import RNNModel
from .nn.fnn import FNNModel
from .distillation import DistillationModel
from .presets import DistillationConfig, ReservoirConfig
from reservoir.training.presets import TrainingConfig


class ModelFactory:
    """Create BaseModel-compatible instances (FNN/RNN/Distillation) from config dictionaries."""

    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        model_type = config.get("type")
        model_cfg = dict(config.get("model", {}) or {})
        raw_training = config.get("training", {})
        if isinstance(raw_training, TrainingConfig):
            training_cfg = raw_training
        else:
            training_dict = dict(raw_training or {})
            allowed = TrainingConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in training_dict.items() if k in allowed}
            training_cfg = TrainingConfig(**filtered)

        if model_type == "fnn":
            if "layer_dims" not in model_cfg:
                raise ValueError("FNN model requires 'layer_dims' list in model config.")
            fnn_cfg = model_cfg
            training = training_cfg

            reservoir_cfg_dict: Dict[str, Any] = config.get("reservoir") or config.get("reservoir_params") or {}
            if reservoir_cfg_dict:
                hidden_dim_override = config.get("hidden_dim")
                if hidden_dim_override is not None:
                    reservoir_cfg_dict = dict(reservoir_cfg_dict)
                    reservoir_cfg_dict.setdefault("n_units", hidden_dim_override)

                teacher_cfg = ReservoirConfig(**reservoir_cfg_dict)
                teacher_cfg.validate(context="distillation.teacher")
                distill_kwargs: Dict[str, Any] = {"teacher": teacher_cfg}

                hidden_layers: tuple[int, ...] = ()
                layer_dims = tuple(int(v) for v in fnn_cfg["layer_dims"])
                if layer_dims:
                    middle = layer_dims[1:-1] if len(layer_dims) >= 2 else []
                    hidden_layers = tuple(int(v) for v in middle if v is not None)
                if hidden_layers:
                    distill_kwargs["student_hidden_layers"] = hidden_layers

                for key in ("learning_rate", "epochs", "batch_size"):
                    distill_kwargs[key] = getattr(training, key)

                distillation_cfg = DistillationConfig(**distill_kwargs)
                distillation_cfg.validate(context="distillation")

                config_input_dim = config.get("input_dim")
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

                print(">> Factory: FNN model operating in Distillation mode (Teacher=Reservoir, Student=FNN).")
                return DistillationModel(distillation_cfg, int(input_dim))

            return FNNModel(fnn_cfg, training)

        if model_type == "rnn":
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

        raise ValueError(
            "Unsupported model_type for FlaxModelFactory. "
            "Expected 'fnn' or 'rnn', got "
            f"{model_type!r}"
        )


class FlaxModelFactory(ModelFactory):
    """Backward-compatible alias for callers expecting FlaxModelFactory.create_model."""

    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        return ModelFactory.create_model(config)
