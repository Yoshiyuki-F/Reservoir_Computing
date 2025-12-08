"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/distillation/factory.py
STEP 4 and 5 (6 is skipped)
Factory for building distillation teacher-student pipelines."""
from __future__ import annotations

from typing import Any, Dict, Optional

from dataclasses import replace
from reservoir.core.identifiers import Model
from reservoir.models.nn.fnn import FNNModel
from reservoir.models.distillation.model import DistillationModel
from reservoir.models.presets import DistillationConfig
from reservoir.models.config import ClassicalReservoirConfig
from reservoir.models.reservoir.classical import ClassicalReservoir
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

        if input_shape is None:
            raise ValueError("input_shape must be provided for distillation (time, features).")
        if len(input_shape) != 2:
            raise ValueError(f"input_shape must be (time, features), got {input_shape}")
        time_steps = int(input_shape[0])

        #1. create teacher
        teacher_node = ClassicalReservoir(
            n_units=projected_input_dim,
            spectral_radius=teacher_cfg.spectral_radius,
            leak_rate=teacher_cfg.leak_rate,
            rc_connectivity=teacher_cfg.rc_connectivity,
            seed=teacher_cfg.seed,
            aggregation_mode=teacher_cfg.aggregation,
        )
        teacher_feature_dim = teacher_node.get_feature_dim(time_steps=time_steps)

        #2. configure student FNN
        student_input_dim = projected_input_dim * time_steps
        h_layers = distillation_config.student.hidden_layers
        hidden_layers = [h_layers] if isinstance(h_layers, int) else list(h_layers or [])
        fnn_cfg_layers = [student_input_dim] + hidden_layers + [teacher_feature_dim]

        #3, create student
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
                "preprocessed": None,  # preprocessing happens upstream
                "projected": (time_steps, projected_input_dim),  # sequence into flatten
                "adapter": (student_input_dim,),  # flattened time-major input to FNN
                "internal": tuple(hidden_layers) if hidden_layers else None,  # hidden layer widths
                "feature": (int(teacher_feature_dim),),  # student output equals teacher feature dim
                "output": (output_dim,),  # readout target size
            },
            "details": {
                "preprocess": "Flatten",
                "agg_mode": "None",
                "student_layers": tuple(hidden_layers) if hidden_layers else None,
                "student_structure": "Flatten -> FNN",
            },
        }
        model.topology_meta = topo_meta
        return model
