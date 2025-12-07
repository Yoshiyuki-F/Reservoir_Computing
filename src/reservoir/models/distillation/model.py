"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/distillation/model.py
Distill a teacher feature extractor into a student feed-forward network.
Implements strict teacher-student distillation: student mimics teacher outputs, labels are ignored.
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp

from reservoir.models.nn.base import BaseModel
from reservoir.training.presets import TrainingConfig
from reservoir.models.sequential import SequentialModel


class DistillationModel(BaseModel):
    """Wrapper that trains a student SequentialModel to mimic a teacher SequentialModel."""

    def __init__(self, teacher: SequentialModel, student: SequentialModel, training_config: TrainingConfig):
        self.teacher = teacher
        self.student = student
        self.training_config = training_config

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return self.student(inputs)

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.__call__(X)

    def train(self, inputs: jnp.ndarray, targets: Any = None, **kwargs: Any) -> Dict[str, Any]:
        # Step A: Teacher forward pass (no gradients back to teacher).
        teacher_features = jax.lax.stop_gradient(self.teacher(inputs))

        # Step B: Adapter forward (flatten) then student engine train.
        if not hasattr(self.student, "layers") or len(self.student.layers) < 2:
            raise ValueError("Distillation student must be SequentialModel with [Flatten, FNN] layers.")

        adapter = self.student.layers[0]
        engine = self.student.layers[1]
        student_inputs = adapter(inputs)

        logs = engine.train(student_inputs, teacher_features)

        # Post-train: compute final distillation loss for logging.
        student_out = engine.predict(student_inputs)
        if student_out.shape != teacher_features.shape:
            raise ValueError(f"Distillation shape mismatch: student {student_out.shape} vs teacher {teacher_features.shape}")
        distill_loss = float(jnp.mean((student_out - teacher_features) ** 2))
        if isinstance(logs, dict):
            logs = dict(logs)
            logs.setdefault("final_loss", distill_loss)
        else:
            logs = {"final_loss": distill_loss}
        return logs

    def evaluate(self, X: jnp.ndarray, y: Any = None) -> Dict[str, float]:
        """
        Distillation evaluation aligns student outputs with teacher features (not labels).
        """
        teacher_features = jax.lax.stop_gradient(self.teacher(X))
        student_out = self.student(X)
        if student_out.shape != teacher_features.shape:
            raise ValueError(f"Distillation shape mismatch: student {student_out.shape} vs teacher {teacher_features.shape}")
        distill_mse = float(jnp.mean((student_out - teacher_features) ** 2))
        return {"distill_mse": distill_mse}

    def get_topology_meta(self) -> Dict[str, Any]:
        return (
            getattr(self, "topology_meta", {})
            or getattr(self.student, "topology_meta", {})
            or getattr(self.teacher, "topology_meta", {})
            or {}
        )
