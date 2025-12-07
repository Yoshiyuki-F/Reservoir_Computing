"""
Shared configuration components for model pipelines (Steps 2-6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from reservoir.core.identifiers import AggregationMode, Preprocessing


@dataclass(frozen=True)
class PreprocessingConfig:
    """Step 2 preprocessing parameters."""

    method: Preprocessing
    poly_degree: int

    def validate(self, context: str = "preprocess") -> "PreprocessingConfig":
        if self.method is None:
            raise ValueError(f"{context}: method is required.")
        if self.poly_degree is None or int(self.poly_degree) < 1:
            raise ValueError(f"{context}: poly_degree must be >=1.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value if isinstance(self.method, Preprocessing) else str(self.method),
            "poly_degree": int(self.poly_degree),
        }


@dataclass(frozen=True)
class ProjectionConfig:
    """Step 3 projection parameters."""

    n_units: int
    input_scale: float
    input_connectivity: float
    bias_scale: float
    seed: int

    def validate(self, context: str = "projection") -> "ProjectionConfig":
        prefix = f"{context}: "
        if int(self.n_units) <= 0:
            raise ValueError(f"{prefix}n_units must be positive.")
        if float(self.input_scale) <= 0:
            raise ValueError(f"{prefix}input_scale must be positive.")
        if not (0.0 < float(self.input_connectivity) <= 1.0):
            raise ValueError(f"{prefix}input_connectivity must be in (0,1].")
        if float(self.bias_scale) < 0:
            raise ValueError(f"{prefix}bias_scale must be non-negative.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_units": int(self.n_units),
            "input_scale": float(self.input_scale),
            "input_connectivity": float(self.input_connectivity),
            "bias_scale": float(self.bias_scale),
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class ClassicalReservoirConfig:
    """Step 5 reservoir dynamics parameters."""

    spectral_radius: float
    leak_rate: float
    rc_connectivity: float
    seed: int
    aggregation: AggregationMode

    def validate(self, context: str = "dynamics") -> "ClassicalReservoirConfig":
        prefix = f"{context}: "
        if float(self.spectral_radius) <= 0:
            raise ValueError(f"{prefix}spectral_radius must be positive.")
        if not (0.0 < float(self.leak_rate) <= 1.0):
            raise ValueError(f"{prefix}leak_rate must be in (0,1].")
        if not (0.0 < float(self.rc_connectivity) <= 1.0):
            raise ValueError(f"{prefix}rc_connectivity must be in (0,1].")
        if self.aggregation is None:
            raise ValueError(f"{prefix}aggregation is required.")
        if not isinstance(self.aggregation, AggregationMode):
            raise TypeError(f"{prefix}aggregation must be AggregationMode, got {type(self.aggregation)}.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "spectral_radius": float(self.spectral_radius),
            "leak_rate": float(self.leak_rate),
            "rc_connectivity": float(self.rc_connectivity),
            "seed": int(self.seed),
            "aggregation": self.aggregation.value,
        }

@dataclass(frozen=True)
class DistillationConfig:
    """Configuration for distilling reservoir dynamics into a Student FNN."""
    """Step 5 distillation fnn dynamics parameters."""

    teacher: ClassicalReservoirConfig
    student_hidden_layers: Tuple[int, ...]

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
