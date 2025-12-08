"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/distillation/model.py
Distill a teacher feature extractor into a student feed-forward network.
Implements strict teacher-student distillation: student mimics teacher outputs, labels are ignored.
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp

from reservoir.models.reservoir.classical import ClassicalReservoir
from reservoir.models.nn.fnn import FNNModel
from reservoir.models.nn.base import BaseModel
from reservoir.training.presets import TrainingConfig


class DistillationModel(BaseModel):
    """
    Distills reservoir dynamics into a student FNN.
    Teacher: ClassicalReservoir (handles aggregation internally; no gradients).
    Student: FNNModel trained to regress onto teacher targets.
    """

    def __init__(
        self,
        teacher: ClassicalReservoir,
        student: FNNModel,
        training_config: TrainingConfig,
    ):
        self.teacher = teacher
        self.student = student
        self.training_config = training_config

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return self.predict(inputs)

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.student.predict(X)

    def _compute_teacher_targets(self, inputs: jnp.ndarray) -> jnp.ndarray:
        teacher_outputs = self.teacher(inputs, return_sequences=False)
        return jax.lax.stop_gradient(teacher_outputs)

    def train(self, inputs: jnp.ndarray, targets: Any = None, **kwargs: Any) -> Dict[str, Any]:
        teacher_targets = self._compute_teacher_targets(inputs)
        student_logs = self.student.train(inputs, teacher_targets, **kwargs) or {}

        student_out = self.student.predict(inputs)
        if student_out.shape != teacher_targets.shape:
            raise ValueError(
                f"Distillation shape mismatch: student {student_out.shape} vs teacher {teacher_targets.shape}"
            )
        distill_mse = float(jnp.mean((student_out - teacher_targets) ** 2))
        logs = dict(student_logs) if isinstance(student_logs, dict) else {}
        logs.setdefault("distill_mse", distill_mse)
        logs.setdefault("final_loss", distill_mse)
        return logs

    def evaluate(self, X: jnp.ndarray, y: Any = None) -> Dict[str, float]:
        """
        Distillation evaluation aligns student outputs with teacher features (not labels).
        """
        teacher_targets = self._compute_teacher_targets(X)
        student_out = self.student.predict(X)
        if student_out.shape != teacher_targets.shape:
            raise ValueError(
                f"Distillation shape mismatch: student {student_out.shape} vs teacher {teacher_targets.shape}"
            )
        distill_mse = float(jnp.mean((student_out - teacher_targets) ** 2))
        student_metrics = self.student.evaluate(X, teacher_targets)
        if isinstance(student_metrics, dict):
            metrics: Dict[str, float] = dict(student_metrics)
        else:
            metrics = {}
        metrics.setdefault("distill_mse", distill_mse)
        return metrics

    def get_topology_meta(self) -> Dict[str, Any]:
        return (
            getattr(self, "topology_meta", {})
            or getattr(self.student, "topology_meta", {})
            or getattr(self.teacher, "topology_meta", {})
            or {}
        )
