"""src/reservoir/models/distillation/config.py"""
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from reservoir.models.reservoir.config import ReservoirConfig


@dataclass(frozen=True)
class DistillationConfig:
    """Configuration for distilling reservoir dynamics into a Student FNN."""

    teacher: ReservoirConfig = field(default_factory=ReservoirConfig)
    student_hidden_layers: Tuple[int, ...] = (10,)

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "teacher": self.teacher.to_dict(),
            "student_hidden_layers": tuple(int(v) for v in self.student_hidden_layers),
        }

    def validate(self, *, context: str = "") -> None:
        prefix = f"{context}: " if context else ""
        # Teacher config may be partially specified at preset load time; defer strict
        # validation of required fields (like n_units) to factory/model assembly.
        if self.teacher.n_units is not None:
            self.teacher.validate(context=f"{prefix}teacher")
        if not self.student_hidden_layers:
            raise ValueError(f"{prefix}student_hidden_layers must contain at least one layer size.")
        if any(width <= 0 for width in self.student_hidden_layers):
            raise ValueError(f"{prefix}student_hidden_layers values must be positive.")
