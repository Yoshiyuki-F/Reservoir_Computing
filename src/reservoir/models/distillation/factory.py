"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/distillation/factory.py
STEP 4 and 5 (6 is skipped)
Factory for building distillation teacher-student pipelines."""
from __future__ import annotations

from typing import Any, Dict, Optional

from dataclasses import replace
from reservoir.core.identifiers import Pipeline
from reservoir.models.distillation.model import DistillationModel
from reservoir.models.nn.factory import NNModelFactory
from reservoir.models.presets import DistillationConfig
from reservoir.models.reservoir.factory import ReservoirFactory
from reservoir.models.sequential import SequentialModel
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
    ) -> DistillationModel:
        distillation_config.teacher.validate(context="distillation.teacher")

        teacher_seq = ReservoirFactory.create_pipeline(
            distillation_config.teacher.to_dict(),
            input_dim=input_dim,
            output_dim=output_dim,
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

        # Student pipeline: Adapter -> FNN (inputs already projected)
        time_steps = input_shape[0] if input_shape else 1
        student_input_dim = input_dim * time_steps
        fnn_cfg_layers = [student_input_dim, *distillation_config.student_hidden_layers, teacher_feature_dim]
        # Force regression for student (it mimics features, not labels)
        student_training = replace(training, classification=False)
        student_model = NNModelFactory.create_fnn({"layer_dims": fnn_cfg_layers}, student_training)
        student_layers: list[Any] = [Flatten(), student_model]

        student_seq = SequentialModel(student_layers)
        student_seq.effective_input_dim = student_input_dim

        model = DistillationModel(teacher=teacher_seq, student=student_seq, training_config=training)

        topo_meta: Dict[str, Any] = {
            "type": Pipeline.FNN_DISTILLATION.value.upper(),
            "shapes": {
                "input": None,
                "preprocessed": None,
                "projected": None,
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
