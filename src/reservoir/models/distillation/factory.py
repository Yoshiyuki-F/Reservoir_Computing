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
from reservoir.layers.adapters import TimeDelayEmbedding  # New import


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
        
        # Check input_shape dimensionality
        # Expect (Batch, Time, Features) -> 3D
        if len(input_shape) < 3:
             # Fallback or error? Assuming 3D for distillation of sequences
             time_steps = 1
             batch_size = int(input_shape[0])
        else:
             batch_size = int(input_shape[0])
             time_steps = int(input_shape[1])

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

        #2. configure student FNN with Time Delay Embedding
        # Default window size 10 (hardcoded for Mackey-Glass Hard Mode per user request)
        # Ideally this should be in config, but we are fixing the pipeline now.
        window_size = 10
        student_input_dim = projected_input_dim * window_size
        h_layers = distillation_config.student.hidden_layers
        hidden_layers = [h_layers] if isinstance(h_layers, int) else list(h_layers or [])

        #3, create student
        student_training = replace(training, classification=False)
        student_model = FNNModel(
            model_config=distillation_config.student,
            training_config=student_training,
            input_dim=student_input_dim,
            output_dim=int(teacher_feature_dim),
        )

        model = DistillationModel(
            teacher=teacher_node,
            student=student_model,
            training_config=student_training,
            student_adapter=TimeDelayEmbedding(window_size=window_size) # Pass adapter to model
        )

        topo_meta: Dict[str, Any] = {
            "type": Model.FNN_DISTILLATION.value.upper(),
            "shapes": {
                "input": input_shape,
                "preprocessed": None,  # preprocessing happens upstream
                "projected": (batch_size, time_steps, projected_input_dim),  # sequence into flatten
                "adapter": (batch_size, student_input_dim),  # flattened time-major input to FNN
                "internal": tuple(hidden_layers) if hidden_layers else None,  # hidden layer widths
                "feature": (batch_size, int(teacher_feature_dim)),  # student output equals teacher feature dim
                "output": (batch_size, output_dim),  # readout target size
            },
            "details": {
                "preprocess": "TDE",
                "agg_mode": "None",
                "student_layers": tuple(hidden_layers) if hidden_layers else None,
                "student_structure": f"TDE(w={window_size}) -> FNN",
            },
        }
        model.topology_meta = topo_meta
        return model
