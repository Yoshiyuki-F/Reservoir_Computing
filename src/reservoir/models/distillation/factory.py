"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/distillation/factory.py
STEP 4 and 5 (6 is skipped)
Factory for building distillation teacher-student pipelines."""
from __future__ import annotations

from typing import Any, Dict, Optional

import jax.numpy as jnp
from dataclasses import replace
from reservoir.core.identifiers import Model
from reservoir.models.nn.fnn import FNNModel
from reservoir.models.distillation.model import DistillationModel
from reservoir.models.presets import DistillationConfig
from reservoir.models.config import ClassicalReservoirConfig
from reservoir.models.reservoir.factory import ReservoirFactory
from reservoir.training.presets import TrainingConfig


class DistillationFactory:
    """Builds teacher (reservoir pipeline) and student (FNN) for distillation."""

    @staticmethod
    def create_model(
        distillation_config: DistillationConfig,
        training: TrainingConfig,
        input_dim: int,
        output_dim: int,
        input_shape: Optional[tuple[int, ...]],
    ) -> DistillationModel:
        teacher_cfg = distillation_config.teacher
        if not isinstance(teacher_cfg, ClassicalReservoirConfig):
            raise TypeError(f"Distillation teacher must be ClassicalReservoirConfig, got {type(teacher_cfg)}.")
        teacher_cfg.validate(context="distillation.teacher")

        projected_input_dim = int(input_dim)
        if projected_input_dim <= 0:
            raise ValueError(f"input_dim must be positive for distillation, got {input_dim}")

        teacher_node = ReservoirFactory.create_node(teacher_cfg, projected_input_dim)
        aggregator = teacher_node.aggregator

        time_steps = input_shape[0] if input_shape else 1
        dummy_states = jnp.zeros((1, time_steps, projected_input_dim), dtype=jnp.float64)
        aggregated = aggregator.transform(dummy_states)
        teacher_feature_dim = int(aggregated.shape[-1]) if aggregated.ndim >= 2 else int(aggregated.size)

        student_input_dim = projected_input_dim * time_steps
        fnn_cfg_layers = (
            student_input_dim,
            *distillation_config.student_hidden_layers,
            teacher_feature_dim,
        )
        student_training = replace(training, classification=False)
        student_model = FNNModel({"layer_dims": fnn_cfg_layers}, student_training)

        model = DistillationModel(
            teacher=teacher_node,
            student=student_model,
            training_config=student_training,
        )

        topo_meta: Dict[str, Any] = {
            "type": Model.FNN_DISTILLATION.value.upper(),
            "shapes": {
                "input": input_shape,
                "preprocessed": None,
                "projected": (time_steps, projected_input_dim) if input_shape else None,
                "internal": (time_steps, projected_input_dim),
                "feature": (int(teacher_feature_dim),),
                "output": (output_dim,),
            },
            "details": {
                "preprocess": None,
                "agg_mode": teacher_cfg.aggregation.value,
                "student_layers": tuple(fnn_cfg_layers[1:-1]) if len(fnn_cfg_layers) > 2 else None,
                "student_structure": "Flatten -> FNN",
            },
        }
        model.topology_meta = topo_meta
        return model
