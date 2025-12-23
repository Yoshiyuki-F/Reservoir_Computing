"""
Shared configuration components for model pipelines (Steps 2-6).
config shouldnt have initial value!
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple, Union, Optional

from reservoir.core.identifiers import AggregationMode, Preprocessing, Model



@dataclass(frozen=True)
class PipelineConfig:
    """
    Canonical pipeline configuration composed of step-specific configs (2-6).
    Exposes explicit, validated fields; no implicit defaults.
    """

    name: str
    model_type: Model
    description: str
    preprocess: PreprocessingConfig
    projection: Optional[ProjectionConfig]
    model: ModelConfig
    readout: ReadoutConfig

    def __post_init__(self) -> None:
        if self.preprocess is None:
            raise ValueError(f"{self.name}: preprocess config is required.")
        if self.model is None:
            raise ValueError(f"{self.name}: model config is required.")

        self.preprocess.validate(context=f"{self.name}.preprocess")
        if self.projection is not None:
            self.projection.validate(context=f"{self.name}.projection")
        if self.readout is not None:
            self.readout.validate(context=f"{self.name}.readout")

        model_cfg = self.model
        if isinstance(model_cfg, DistillationConfig):
            model_cfg.validate(context=f"{self.name}.model")
        elif hasattr(model_cfg, "validate"):
            model_cfg.validate(context=f"{self.name}.model")

    # Legacy-friendly aliases (SSOT remains the explicit fields above)
    @property
    def config(self) -> Union[ClassicalReservoirConfig, DistillationConfig]:
        return self.model

    @property
    def preprocess_config(self) -> PreprocessingConfig:
        return self.preprocess

    @property
    def projection_config(self) -> ProjectionConfig:
        return self.projection

    @property
    def params(self) -> Dict[str, Any]:
        return {}

    @property
    def reservoir(self) -> Optional[ClassicalReservoirConfig]:
        """Expose reservoir configs for consumers expecting the legacy attribute."""
        return self.model if isinstance(self.model, ClassicalReservoirConfig) else None

    @property
    def distillation(self) -> Optional[DistillationConfig]:
        """Expose distillation configs for consumers expecting the legacy attribute."""
        return self.model if isinstance(self.model, DistillationConfig) else None

    def to_params(self) -> Dict[str, Any]:
        """Flatten pipeline configuration into a serializable dictionary."""
        merged: Dict[str, Any] = {}

        model_cfg = self.model
        if isinstance(model_cfg, DistillationConfig):
            merged.update(model_cfg.teacher.to_dict())
            merged["student.hidden_layers"] = tuple(int(v) for v in model_cfg.student.hidden_layers)
        elif hasattr(model_cfg, "to_dict"):
            merged.update(model_cfg.to_dict())
        else:
            merged.update(asdict(model_cfg))

        merged.update(self.preprocess.to_dict())
        if self.projection is not None:
            merged.update(self.projection.to_dict())
        merged.update(self.readout.to_dict())

        return merged

    def __getattr__(self, item: str) -> Any:
        """
        Provide passthrough access to underlying model config for convenience
        (e.g., preset.leak_rate).
        """
        model_cfg = object.__getattribute__(self, "model")
        if hasattr(model_cfg, item):
            return getattr(model_cfg, item)
        if isinstance(model_cfg, DistillationConfig) and hasattr(model_cfg.teacher, item):
            return getattr(model_cfg.teacher, item)
        raise AttributeError(f"{item} not found in PipelineConfig or underlying model config.")




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
    student: FNNConfig

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "teacher": self.teacher.to_dict(),
            "student.hidden_layers": tuple(int(v) for v in self.student.hidden_layers),
        }

    def validate(self, *, context: str = "") -> None:
        prefix = f"{context}: " if context else ""
        self.teacher.validate(context=f"{prefix}teacher")
        if not self.student.hidden_layers:
            raise ValueError(f"{prefix}student.hidden_layers must contain at least one layer size.")
        if any(width < 0 for width in self.student.hidden_layers):
            raise ValueError(f"{prefix}student.hidden_layers values must be non negative.")


@dataclass(frozen=True)
class FNNConfig:
    hidden_layers: Optional[Tuple[int, ...]]

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_layers": tuple(int(v) for v in (self.hidden_layers or ())),
        }

    def validate(self, *, context: str = "") -> None:
        prefix = f"{context}: " if context else ""
        layers = self.hidden_layers or ()
        if any(width < 0 for width in layers):
            raise ValueError(f"{prefix}hidden_layers values must be non-negative.")


@dataclass(frozen=True)
class RidgeReadoutConfig:
    """Step 7 readout configuration (structure/defaults)."""
    init_lambda: float
    use_intercept: bool

    def validate(self, context: str = "ridgereadout") -> "RidgeReadoutConfig":
        if float(self.init_lambda) <= 0:
            raise ValueError(f"{context}: init_lambda must be positive.")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"init_lambda": float(self.init_lambda), "use_intercept": bool(self.use_intercept)}

@dataclass(frozen=True)
class FNNReadoutConfig:
    """Step 7 readout configuration (structure/fnn)."""
    hidden_layers: Optional[Tuple[int, ...]]

    def validate(self, context: str = "fnnreadout") -> "FNNReadoutConfig":
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"hidden_layers": tuple(self.hidden_layers or ())}


ModelConfig = Union[ClassicalReservoirConfig, DistillationConfig, FNNConfig]
ReadoutConfig = Union[RidgeReadoutConfig, None]
