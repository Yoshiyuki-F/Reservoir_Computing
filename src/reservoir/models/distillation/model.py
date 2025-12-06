"""
Distill a teacher feature extractor into a student feed-forward network.
"""

from __future__ import annotations

from typing import Any, Dict

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
        teacher_features = self.teacher(inputs)
        # Force MSE-based distillation by disabling classification loss on the student model
        # and any trainable layers inside the sequential container.
        if hasattr(self.student, "classification"):
            try:
                self.student.classification = False
            except Exception:
                pass
        if hasattr(self.student, "layers"):
            for layer in getattr(self.student, "layers"):
                if hasattr(layer, "classification"):
                    try:
                        layer.classification = False
                    except Exception:
                        pass
        return self.student.train(inputs, teacher_features)

    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        y_arr = jnp.asarray(y)
        if preds.shape != y_arr.shape:
            raise ValueError(f"Shape mismatch for evaluation: preds {preds.shape}, y {y_arr.shape}")
        mse = float(jnp.mean((preds - y_arr) ** 2))
        return {"mse": mse}

    def get_topology_meta(self) -> Dict[str, Any]:
        return (
            getattr(self, "topology_meta", {})
            or getattr(self.student, "topology_meta", {})
            or getattr(self.teacher, "topology_meta", {})
            or {}
        )
