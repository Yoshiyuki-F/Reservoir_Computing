"""src/reservoir/models/distillation/config.py"""
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from reservoir.models.reservoir.classical.config import ClassicalReservoirConfig


@dataclass(frozen=True)
class DistillationConfig:
    """Configuration for distilling reservoir dynamics into a Student FNN."""

    teacher: ClassicalReservoirConfig = field(default_factory=ClassicalReservoirConfig)
    student_hidden_layers: Tuple[int, ...] = (300,)

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "teacher": self.teacher.to_dict(),
            "student_hidden_layers": tuple(int(v) for v in self.student_hidden_layers),
        }

    def validate(self, *, context: str = "") -> None:
        prefix = f"{context}: " if context else ""
        self.teacher.validate(context=f"{prefix}teacher")
        if not self.student_hidden_layers:
            raise ValueError(f"{prefix}student_hidden_layers must contain at least one layer size.")
        if any(width <= 0 for width in self.student_hidden_layers):
            raise ValueError(f"{prefix}student_hidden_layers values must be positive.")
