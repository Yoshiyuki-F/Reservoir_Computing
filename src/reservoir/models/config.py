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
    """Step 5 and 6 reservoir dynamics parameters."""

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
    """Step 5 and 6 distillation fnn dynamics parameters."""

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
    """FNN configuration with optional sliding window for time series."""
    hidden_layers: Optional[Tuple[int, ...]]
    window_size: Optional[int] = None  # None = Flatten, int = TimeDelayEmbedding(K)

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        result = {
            "hidden_layers": tuple(int(v) for v in (self.hidden_layers or ())),
        }
        if self.window_size is not None:
            result["window_size"] = int(self.window_size)
        return result

    def validate(self, *, context: str = "") -> None:
        prefix = f"{context}: " if context else ""
        layers = self.hidden_layers or ()
        if any(width < 0 for width in layers):
            raise ValueError(f"{prefix}hidden_layers values must be non-negative.")
        if self.window_size is not None and int(self.window_size) < 1:
            raise ValueError(f"{prefix}window_size must be >= 1.")







@dataclass(frozen=True)
class PassthroughConfig:
    """Configuration for passthrough model that skips dynamics (Step 5)."""
    aggregation: AggregationMode

    def validate(self, context: str = "passthrough") -> "PassthroughConfig":
        if not isinstance(self.aggregation, AggregationMode):
            raise TypeError(f"{context}: aggregation must be AggregationMode, got {type(self.aggregation)}.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregation": self.aggregation.value,
        }


@dataclass(frozen=True)
class QuantumReservoirConfig:
    """Step 5 Quantum Reservoir dynamics parameters.
    
    Note: n_qubits is NOT stored here - it is derived from projection.n_units at runtime.
    
    measurement_basis options:
        - 'Z': 1st moment only (n_qubits features)
        - 'ZZ': 2-point correlations only (n_qubits*(n_qubits-1)/2 features)
        - 'Z+ZZ': 1st moment + 2-point correlations (n_qubits + n_qubits*(n_qubits-1)/2 features)
                  For 4 qubits: 4 + 6 = 10 features
    """
    
    n_layers: int                    # Number of variational layers
    seed: int                        # Random seed for fixed parameters
    aggregation: AggregationMode     # How to aggregate time steps
    input_scaling: float             # Scaling factor for input encoding (typically 2π)
    feedback_scale: float            # Scaling factor for state feedback (e.g. 0.1)
    measurement_basis: str           # 'Z', 'ZZ', 'Z+ZZ' for correlation measurements
    encoding_strategy: str   # 'Rx', 'Ry', 'Rz', 'IQP'
    noise_type: str        # 'clean', 'depolarizing', 'damping'
    noise_prob: float          # Probability of noise (0.0 to 1.0) Default Value is forbidden, should be in the prefix.py

    def validate(self, context: str = "quantum_reservoir") -> "QuantumReservoirConfig":
        prefix = f"{context}: "
        if int(self.n_layers) <= 0:
            raise ValueError(f"{prefix}n_layers must be positive.")
        if float(self.input_scaling) <= 0:
            raise ValueError(f"{prefix}input_scaling must be positive.")
        if float(self.feedback_scale) < 0:
            raise ValueError(f"{prefix}feedback_scale must be non-negative.")
        if not isinstance(self.aggregation, AggregationMode):
            raise TypeError(f"{prefix}aggregation must be AggregationMode, got {type(self.aggregation)}.")
        # Validate measurement_basis
        valid_bases = ("Z", "ZZ", "Z+ZZ")
        if self.measurement_basis not in valid_bases:
            raise ValueError(f"{prefix}measurement_basis must be one of {valid_bases}.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_layers": int(self.n_layers),
            "seed": int(self.seed),
            "input_scaling": float(self.input_scaling),
            "feedback_scale": float(self.feedback_scale),
            "aggregation": self.aggregation.value,
            "measurement_basis": str(self.measurement_basis),
            "encoding_strategy": str(self.encoding_strategy),
            "noise_type": str(self.noise_type),
            "noise_prob": float(self.noise_prob),
        }


@dataclass(frozen=True)
class RidgeReadoutConfig:
    """Step 7 readout configuration (structure/defaults)."""
    init_lambda: float
    use_intercept: bool
    lambda_candidates: Optional[Tuple[float, ...]] = None

    def validate(self, context: str = "ridgereadout") -> "RidgeReadoutConfig":
        if float(self.init_lambda) <= 0:
            raise ValueError(f"{context}: init_lambda must be positive.")
        if self.lambda_candidates is not None:
            if any(float(lam) <= 0.0 for lam in self.lambda_candidates):
                raise ValueError(f"{context}: lambda_candidates must contain only positive values.")
        return self

    def to_dict(self) -> Dict[str, Any]:
        result = {"init_lambda": float(self.init_lambda), "use_intercept": bool(self.use_intercept)}
        if self.lambda_candidates is not None:
            result["lambda_candidates"] = [float(v) for v in self.lambda_candidates]
        return result

@dataclass(frozen=True)
class FNNReadoutConfig:
    """Step 7 readout configuration (structure/fnn)."""
    hidden_layers: Optional[Tuple[int, ...]]

    def validate(self, context: str = "fnnreadout") -> "FNNReadoutConfig":
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"hidden_layers": tuple(self.hidden_layers or ())}


ModelConfig = Union[ClassicalReservoirConfig, DistillationConfig, FNNConfig, PassthroughConfig, QuantumReservoirConfig]
ReadoutConfig = Union[RidgeReadoutConfig, FNNReadoutConfig, None]
