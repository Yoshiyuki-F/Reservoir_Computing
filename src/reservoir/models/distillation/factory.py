"""src/reservoir/models/distillation/factory.py"""
from typing import Any, Dict

from reservoir.training.presets import TrainingConfig
from reservoir.models.distillation.config import DistillationConfig
from reservoir.models.reservoir.classical.config import ClassicalReservoirConfig
from reservoir.models.reservoir.factory import ReservoirFactory
from reservoir.models.reservoir.model import ReservoirModel
from .model import DistillationModel


class DistillationFactory:
    """Factory specialized in assembling Teacher-Student distillation models."""

    @staticmethod
    def create(
        fnn_cfg: Dict[str, Any],
        training_cfg: TrainingConfig,
        full_config: Dict[str, Any]
    ) -> DistillationModel:
        """
        Builds a DistillationModel by constructing a Reservoir Teacher 
        and configuring a FNN Student.
        """
        # 1. Prepare Teacher Configuration
        reservoir_cfg_dict = full_config.get("reservoir") or full_config.get("reservoir_params")
        if not reservoir_cfg_dict:
            raise ValueError("Distillation requires 'reservoir' or 'reservoir_params' in config.")

        # Override hidden dim if specified globally
        hidden_dim_override = full_config.get("hidden_dim")
        if hidden_dim_override is not None:
            reservoir_cfg_dict = dict(reservoir_cfg_dict)
            reservoir_cfg_dict.setdefault("n_units", hidden_dim_override)

        teacher_cfg = ClassicalReservoirConfig(**reservoir_cfg_dict)
        teacher_cfg.validate(context="distillation.teacher")

        # 2. Prepare Student Configuration
        layer_dims = [int(v) for v in (fnn_cfg.get("layer_dims") or [])]
        student_hidden = DistillationFactory._resolve_student_hidden(
            layer_dims=layer_dims,
            full_config=full_config,
        )

        distill_cfg = DistillationConfig(
            teacher=teacher_cfg,
            student_hidden_layers=student_hidden
        )

        # 3. Validate Dimensions
        input_dim = DistillationFactory._validate_input_dim(full_config, layer_dims)

        # 4. Build Teacher Node & Model
        teacher_node = ReservoirFactory.create_node(teacher_cfg, input_dim)
        teacher_model = ReservoirModel(
            reservoir=teacher_node, 
            preprocess=None, 
            readout_mode=teacher_cfg.state_aggregation or "mean"
        )

        print(">> DistillationFactory: Assembled Teacher (Reservoir) + Student (FNN).")
        return DistillationModel(
            config=distill_cfg,
            training_config=training_cfg,
            input_dim=int(input_dim),
            teacher_model=teacher_model
        )

    @staticmethod
    def _validate_input_dim(full_config: Dict[str, Any], layer_dims: list) -> int:
        config_input_dim = full_config.get("input_dim")
        inferred_input_dim = layer_dims[0] if layer_dims else None

        if config_input_dim is not None and inferred_input_dim is not None:
            if int(config_input_dim) != int(inferred_input_dim):
                raise ValueError(
                    f"Input dim mismatch: config={config_input_dim} vs FNN layer={inferred_input_dim}"
                )
        
        input_dim = config_input_dim or inferred_input_dim
        if input_dim is None:
            raise ValueError("Could not determine input_dim for DistillationModel.")
            
        return int(input_dim)

    @staticmethod
    def _resolve_student_hidden(layer_dims: list[int], full_config: Dict[str, Any]) -> tuple[int, ...]:
        """
        Resolve student hidden layers.
        Accepts either explicit hidden list (--nn-hidden / student_hidden_layers) or
        FNN layer_dims that may already include input/output dims.
        """
        # Highest priority: explicit student hidden provided outside layer_dims
        explicit_hidden = full_config.get("student_hidden") or full_config.get("student_hidden_layers") or full_config.get("nn_hidden")
        if explicit_hidden:
            return tuple(int(v) for v in explicit_hidden)

        if layer_dims:
            # If layer_dims length suggests it includes input/output, strip them; otherwise treat as hidden list.
            if len(layer_dims) > 2:
                return tuple(layer_dims[1:-1])
            return tuple(layer_dims)

        # Fallback to default from DistillationConfig
        return DistillationConfig().student_hidden_layers
