"""
Shared configuration components for model pipelines (Steps 2-6).
config shouldnt have initial value!
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple, Union, Optional

from reservoir.core.identifiers import AggregationMode, Model



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
class RawConfig:
    """Step 2 parameters for Raw (no preprocessing)."""

    def validate(self, context: str = "raw") -> "RawConfig":
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"method": "raw"}


@dataclass(frozen=True)
class StandardScalerConfig:
    """Step 2 parameters for Standard Scaler (mean removal and variance scaling)."""

    def validate(self, context: str = "standard_scaler") -> "StandardScalerConfig":
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"method": "standard_scaler"}


@dataclass(frozen=True)
class CustomRangeScalerConfig:
    """Step 2 parameters for Custom Range Scaler."""
    scale: float
    centering: bool = False

    def validate(self, context: str = "custom_range_scaler") -> "CustomRangeScalerConfig":
        if float(self.scale) == 0:
            raise ValueError(f"{context}: scale must be non-zero.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"method": "custom_range_scaler", "scale": float(self.scale), "centering": self.centering}

PreprocessingConfig = Union[RawConfig, StandardScalerConfig, CustomRangeScalerConfig]


@dataclass(frozen=True)
class RandomProjectionConfig:
    """Step 3 parameters for Random Projection."""
    n_units: int
    input_scale: float
    input_connectivity: float
    bias_scale: float
    seed: int
    
    def validate(self, context: str = "random_projection") -> "RandomProjectionConfig":
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
            "type": "random",
            "n_units": int(self.n_units),
            "input_scale": float(self.input_scale),
            "input_connectivity": float(self.input_connectivity),
            "bias_scale": float(self.bias_scale),
            "seed": int(self.seed),
        }

@dataclass(frozen=True)
class CenterCropProjectionConfig:
    """Step 3 parameters for Center Crop Projection (3D input only)."""
    n_units: int
    
    def validate(self, context: str = "center_crop") -> "CenterCropProjectionConfig":
        prefix = f"{context}: "
        if int(self.n_units) <= 0:
            raise ValueError(f"{prefix}n_units must be positive.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "center_crop",
            "n_units": int(self.n_units),
        }


@dataclass(frozen=True)
class ResizeProjectionConfig:
    """Step 3 parameters for Resize (Interpolation) Projection."""
    n_units: int

    def validate(self, context: str = "resize_projection") -> "ResizeProjectionConfig":
        prefix = f"{context}: "
        if int(self.n_units) <= 0:
            raise ValueError(f"{prefix}n_units must be positive.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "resize",
            "n_units": int(self.n_units),
        }

@dataclass(frozen=True)
class PolynomialProjectionConfig:
    """Step 3 parameters for Polynomial Projection (feature expansion)."""
    degree: int
    include_bias: bool

    def validate(self, context: str = "polynomial_projection") -> "PolynomialProjectionConfig":
        if self.degree is None or int(self.degree) < 1:
            raise ValueError(f"{context}: degree must be >=1.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"type": "polynomial", "degree": int(self.degree), "include_bias": self.include_bias}


@dataclass(frozen=True)
class PCAProjectionConfig:
    """Step 3 parameters for PCA Projection (dimensionality reduction).
    
    Note: PCA assumes StandardScaler preprocessing (zero mean, unit variance).
    
    Args:
        n_units: Number of principal components to keep.
        input_scaler: Scalar to multiply output after projection.
    """
    n_units: int
    input_scaler: float

    def validate(self, context: str = "pca_projection") -> "PCAProjectionConfig":
        if self.n_units is None or int(self.n_units) < 1:
            raise ValueError(f"{context}: n_units must be >=1.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"type": "pca", "n_units": int(self.n_units), "input_scaler": float(self.input_scaler)}


@dataclass(frozen=True)
class CoherentDriveProjectionConfig:
    """Step 3 parameters for Coherent Drive Projection.
    
    Uses arcsin transformation for amplitude-based quantum encoding.
    Non-periodic, suitable for trending time series like Mackey-Glass.
    
    θ = 2 * arcsin(clip(linear_proj(x), -1, 1))
    """
    n_units: int
    input_scale: float
    input_connectivity: float
    bias_scale: float
    seed: int
    
    def validate(self, context: str = "coherent_drive") -> "CoherentDriveProjectionConfig":
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
            "type": "coherent_drive",
            "n_units": int(self.n_units),
            "input_scale": float(self.input_scale),
            "input_connectivity": float(self.input_connectivity),
            "bias_scale": float(self.bias_scale),
            "seed": int(self.seed),
        }


ProjectionConfig = Union[RandomProjectionConfig, CenterCropProjectionConfig, ResizeProjectionConfig, PolynomialProjectionConfig, PCAProjectionConfig, CoherentDriveProjectionConfig]





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
    leak_rate: float          # Leaky integrator rate (alpha) for Li-ESN dynamics
    feedback_scale: float     # Feedback injection scale (gamma). 0.0 = no feedback
    measurement_basis: str  # 'Z', 'ZZ', 'Z+ZZ' for correlation measurements
    encoding_strategy: str   # 'Rx', 'Ry', 'Rz', 'IQP'
    noise_type: str        # 'clean', 'depolarizing', 'damping'
    noise_prob: float          # Probability of noise (0.0 to 1.0)
    readout_error: float     # Readout error probability (0.0 to 1.0)
    n_trajectories: int      # Number of trajectories for Monte Carlo simulation (0 = Density Matrix)
    use_remat: bool          # Use gradient checkpointing (rematerialization)
    use_reuploading: bool    # Use data re-uploading strategy
    precision: str           # "complex64" or "complex128"

    def validate(self, context: str = "quantum_reservoir") -> "QuantumReservoirConfig":
        prefix = f"{context}: "
        if int(self.n_layers) <= 0:
            raise ValueError(f"{prefix}n_layers must be positive.")
        if not (0.0 < float(self.leak_rate) <= 1.0):
             raise ValueError(f"{prefix}leak_rate must be in (0,1].")
        if not isinstance(self.aggregation, AggregationMode):
            raise TypeError(f"{prefix}aggregation must be AggregationMode, got {type(self.aggregation)}.")
        # Validate measurement_basis
        valid_bases = ("Z", "ZZ", "Z+ZZ")
        if self.measurement_basis not in valid_bases:
            raise ValueError(f"{prefix}measurement_basis must be one of {valid_bases}.")
        if float(self.readout_error) < 0.0 or float(self.readout_error) > 1.0:
            raise ValueError(f"{prefix}readout_error must be in [0, 1].")
        if int(self.n_trajectories) < 0:
             raise ValueError(f"{prefix}n_trajectories must be non-negative.")
        valid_precisions = ("complex64", "complex128")
        if self.precision not in valid_precisions:
            raise ValueError(f"{prefix}precision must be one of {valid_precisions}.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_layers": int(self.n_layers),
            "seed": int(self.seed),
            "leak_rate": float(self.leak_rate),
            "aggregation": self.aggregation.value,
            "measurement_basis": str(self.measurement_basis),
            "encoding_strategy": str(self.encoding_strategy),
            "noise_type": str(self.noise_type),
            "noise_prob": float(self.noise_prob),
            "readout_error": float(self.readout_error),
            "n_trajectories": int(self.n_trajectories),
            "use_remat": bool(self.use_remat),
            "use_reuploading": bool(self.use_reuploading),
            "precision": str(self.precision),
        }


@dataclass(frozen=True)
class RidgeReadoutConfig:
    """Step 7 readout configuration (structure/defaults)."""
    use_intercept: bool
    lambda_candidates: Optional[Tuple[float, ...]] = None

    def validate(self, context: str = "ridgereadout") -> "RidgeReadoutConfig":
        if self.lambda_candidates is not None:
            if any(float(lam) <= 0.0 for lam in self.lambda_candidates):
                raise ValueError(f"{context}: lambda_candidates must contain only positive values.")
        return self

    def to_dict(self) -> Dict[str, Any]:
        result = {"use_intercept": bool(self.use_intercept)}
        if self.lambda_candidates is not None:
            result["lambda_candidates"] = [float(v) for v in self.lambda_candidates]
        return result



@dataclass(frozen=True)
class PolyRidgeReadoutConfig:
    """Step 7 poly readout configuration (structure/defaults)."""

    use_intercept: bool
    lambda_candidates: Optional[Tuple[float, ...]]
    degree: int
    mode: Literal["full", "square_only"]

    def validate(self, context: str = "polyridgereadout") -> "PolyRidgeReadoutConfig":
        if self.lambda_candidates is not None:
            if any(float(lam) <= 0.0 for lam in self.lambda_candidates):
                raise ValueError(f"{context}: lambda_candidates must contain only positive values.")
        if int(self.degree) < 2:
            raise ValueError(f"{context}: degree must be >= 2.")
        if self.mode not in ("full", "square_only"):
            raise ValueError(f"{context}: mode must be 'full' or 'square_only', got '{self.mode}'.")
        return self

    def to_dict(self) -> Dict[str, Any]:
        result = {"use_intercept": bool(self.use_intercept), "degree": int(self.degree), "mode": str(self.mode)}
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
ReadoutConfig = Union[RidgeReadoutConfig, PolyRidgeReadoutConfig, FNNReadoutConfig, None]
