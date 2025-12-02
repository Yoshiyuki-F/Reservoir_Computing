"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/factories.py
Factories to build BaseModel-compatible wrappers for Flax modules."""

from __future__ import annotations

from typing import Any, Dict

from .nn.base import BaseModel
from .nn.rnn import RNNModel
from .nn.config import FNNModelConfig, SimpleRNNConfig
from .distillation import DistillationModel
from .presets import DistillationConfig, ReservoirConfig


class ModelFactory:
    """Create BaseModel-compatible instances (FNN/RNN/Distillation) from config dictionaries."""

    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        model_type = config.get("type")
        model_cfg = dict(config.get("model", {}) or {})
        training_cfg = config.get("training", {})

        if model_type == "fnn":
            valid_fnn_fields = set(FNNModelConfig.model_fields.keys())
            fnn_model_cfg = {key: value for key, value in model_cfg.items() if key in valid_fnn_fields}
            ignored_fields = set(model_cfg.keys()) - valid_fnn_fields
            if ignored_fields:
                print(f"ModelFactory: Ignoring unsupported FNN config keys: {sorted(ignored_fields)}")

            fnn_cfg = FNNModelConfig(**fnn_model_cfg)
            reservoir_cfg_dict: Dict[str, Any] = (
                config.get("reservoir")
                or config.get("reservoir_params")
                or {}
            )

            if not reservoir_cfg_dict:
                raise ValueError("FNN mode implies distillation; provide a reservoir teacher configuration.")

            hidden_dim_override = config.get("hidden_dim")
            if hidden_dim_override is not None:
                reservoir_cfg_dict = dict(reservoir_cfg_dict)
                reservoir_cfg_dict.setdefault("n_units", hidden_dim_override)

            teacher_cfg = ReservoirConfig(**reservoir_cfg_dict)
            teacher_cfg.validate(context="distillation.teacher")
            distill_kwargs: Dict[str, Any] = {"teacher": teacher_cfg}

            hidden_layers: tuple[int, ...] = ()
            if fnn_cfg.layer_dims:
                middle = fnn_cfg.layer_dims[1:-1] if len(fnn_cfg.layer_dims) >= 2 else []
                hidden_layers = tuple(int(v) for v in middle if v is not None)
            if hidden_layers:
                distill_kwargs["student_hidden_layers"] = hidden_layers

            param_types = {
                "learning_rate": float,
                "epochs": int,
                "batch_size": int,
            }
            for key, cast_fn in param_types.items():
                if key in training_cfg:
                    distill_kwargs[key] = cast_fn(training_cfg[key])

            distillation_cfg = DistillationConfig(**distill_kwargs)
            distillation_cfg.validate(context="distillation")

            config_input_dim = config.get("input_dim")
            inferred_input_dim = fnn_cfg.input_dim if fnn_cfg.layer_dims else None

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


class FlaxModelFactory(ModelFactory):
    """Backward-compatible alias for callers expecting FlaxModelFactory.create_model."""

    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        return ModelFactory.create_model(config)
