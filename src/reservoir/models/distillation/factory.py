"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/distillation/factory.py
Factory for building distillation teacher-student pipelines."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from reservoir.core.identifiers import Pipeline
from reservoir.models.distillation.model import DistillationModel
from reservoir.models.nn.factory import NNModelFactory
from reservoir.models.presets import DistillationConfig
from reservoir.models.reservoir.factory import ReservoirFactory
from reservoir.models.sequential import SequentialModel
from reservoir.layers.preprocessing import FeatureScaler
from reservoir.layers.projection import InputProjection
from reservoir.layers.adapters import Flatten
from reservoir.training.presets import TrainingConfig


class DistillationFactory:
    """Builds teacher (reservoir pipeline) and student (FNN) for distillation."""

    @staticmethod
    def create_model(
        distillation_config: DistillationConfig,
        *,
        training: TrainingConfig,
        input_dim: int,
        output_dim: int,
        input_shape: Optional[tuple[int, ...]],
        use_preprocessing: bool,
    ) -> DistillationModel:
        distillation_config.teacher.validate(context="distillation.teacher")

        teacher_seq = ReservoirFactory.create_pipeline(
            distillation_config.teacher.to_dict(),
            input_dim=input_dim,
            output_dim=output_dim,
            use_preprocessing=use_preprocessing,
            input_shape=input_shape,
            pipeline=Pipeline.CLASSICAL_RESERVOIR,
        )

        teacher_meta = getattr(teacher_seq, "topology_meta", {}) or {}
        teacher_shapes = teacher_meta.get("shapes", {}) or {}
        teacher_feature_shape = teacher_shapes.get("feature")
        if teacher_feature_shape is None:
            teacher_feature_dim = int(distillation_config.teacher.n_units)
        elif isinstance(teacher_feature_shape, tuple):
            teacher_feature_dim = int(teacher_feature_shape[0])
        else:
            teacher_feature_dim = int(teacher_feature_shape)

        # Student pipeline: Preprocess -> Projection -> Flatten -> FNN
        student_layers: list[Any] = []
        if use_preprocessing:
            student_layers.append(FeatureScaler())

        projection = InputProjection(
            input_dim=input_dim,
            output_dim=distillation_config.teacher.n_units,
            input_scale=distillation_config.teacher.input_scale,
            input_connectivity=distillation_config.teacher.input_connectivity,
            bias_scale=distillation_config.teacher.bias_scale,
            seed=distillation_config.teacher.seed or 0,
        )
        student_layers.append(projection)

        flatten = Flatten()
        student_layers.append(flatten)

        time_steps = input_shape[0] if input_shape else 1
        student_input_dim = distillation_config.teacher.n_units * time_steps
        fnn_cfg_layers = [student_input_dim, *distillation_config.student_hidden_layers, teacher_feature_dim]
        student_model = NNModelFactory.create_fnn({"layer_dims": fnn_cfg_layers}, training)
        student_layers.append(student_model)

        student_seq = SequentialModel(student_layers)
        student_seq.effective_input_dim = student_input_dim

        model = DistillationModel(teacher=teacher_seq, student=student_seq, training_config=training)

        agg_mode = teacher_meta.get("details", {}).get("agg_mode")
        topo_meta: Dict[str, Any] = {
            "type": Pipeline.FNN_DISTILLATION.value.upper(),
            "shapes": {
                "input": input_shape or (1, input_dim),
                "preprocessed": teacher_shapes.get("preprocessed") or teacher_shapes.get("input"),
                "projected": teacher_shapes.get("projected") or (time_steps, distillation_config.teacher.n_units),
                "adapter": (student_input_dim,),
                "internal": (teacher_feature_dim,),
                "feature": (teacher_feature_dim,),
                "output": (output_dim,),
            },
            "details": {
                "preprocess": teacher_meta.get("details", {}).get("preprocess"),
                "agg_mode": None,
                "student_layers": fnn_cfg_layers[1:-1] if len(fnn_cfg_layers) > 2 else None,
                "student_structure": "Proj -> Flatten -> FNN",
            },
        }
        model.topology_meta = topo_meta
        return model
