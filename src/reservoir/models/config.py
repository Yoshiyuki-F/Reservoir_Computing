"""
Shared configuration components for model pipelines (Steps 2-6).
config shouldnt have initial value!
"""

from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Literal, TYPE_CHECKING

from reservoir.layers.aggregation import AggregationMode

if TYPE_CHECKING:
    from reservoir.models.identifiers import Model
    from reservoir.core.types import ConfigDict

# -----------------------------------------------------------------------------
# Base Config ABCs
# -----------------------------------------------------------------------------

class BaseConfig(ABC):
    """Abstract base class for all pipeline configuration components."""

    @abstractmethod
    def validate(self, context: str = "") -> BaseConfig:
        """Validate the configuration parameters."""

    @abstractmethod
    def to_dict(self) -> ConfigDict:
        """Convert the configuration to a dictionary block."""


class PreprocessingConfig(BaseConfig):
    """Base class for Step 2 preprocessing configurations."""
    def validate(self, context: str = "") -> PreprocessingConfig:
        return self
        
    def to_dict(self) -> ConfigDict:
        return {}


class ProjectionConfig(BaseConfig):
    """Base class for Step 3 projection configurations."""
    def validate(self, context: str = "") -> ProjectionConfig:
        return self
        
    def to_dict(self) -> ConfigDict:
        return {}


class ModelConfig(BaseConfig):
    """Base class for Step 5 model dynamics configurations."""
    def validate(self, context: str = "") -> ModelConfig:
        return self
        
    def to_dict(self) -> ConfigDict:
        return {}


class ReadoutConfig(BaseConfig):
    """Base class for Step 7 readout configurations."""
    def validate(self, context: str = "") -> ReadoutConfig:
        return self
        
    def to_dict(self) -> ConfigDict:
        return {}




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
    projection: ProjectionConfig | None
    model: ModelConfig
    readout: ReadoutConfig | None

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

        self.model.validate(context=f"{self.name}.model")

    # Legacy-friendly aliases (SSOT remains the explicit fields above)
    @property
    def config(self) -> ModelConfig:
        return self.model

    @property
    def preprocess_config(self) -> PreprocessingConfig:
        return self.preprocess

    @property
    def projection_config(self) -> ProjectionConfig | None:
        return self.projection

    @property
    def params(self) -> ConfigDict:
        return {}

    @property
    def reservoir(self) -> ClassicalReservoirConfig | None:
        """Expose reservoir configs for consumers expecting the legacy attribute."""
        return self.model if isinstance(self.model, ClassicalReservoirConfig) else None

    @property
    def distillation(self) -> DistillationConfig | None:
        """Expose distillation configs for consumers expecting the legacy attribute."""
        return self.model if isinstance(self.model, DistillationConfig) else None

    def to_params(self) -> ConfigDict:
        """Flatten pipeline configuration into a serializable dictionary."""
        merged: ConfigDict = {}

        merged.update(self.model.to_dict())

        merged.update(self.preprocess.to_dict())
        if self.projection is not None:
            merged.update(self.projection.to_dict())
        if self.readout is not None:
            merged.update(self.readout.to_dict())

        return merged

    def __getattr__(self, item: str):
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
class RawConfig(PreprocessingConfig):
    """Step 2 parameters for Raw (no preprocessing)."""

    def validate(self, context: str = "raw") -> RawConfig:
        _ = context
        return self

    def to_dict(self) -> ConfigDict:
        return {"method": "raw"}


@dataclass(frozen=True)
class StandardScalerConfig(PreprocessingConfig):
    """Step 2 parameters for Standard Scaler (mean removal and variance scaling)."""

    def validate(self, context: str = "standard_scaler") -> StandardScalerConfig:
        _ = context
        return self

    def to_dict(self) -> ConfigDict:
        return {"method": "standard_scaler"}


@dataclass(frozen=True)
class CustomRangeScalerConfig(PreprocessingConfig):
    """Step 2 parameters for Custom Range Scaler."""
    input_scale: float
    centering: bool

    def validate(self, context: str = "custom_range_scaler") -> CustomRangeScalerConfig:
        if float(self.input_scale) == 0:
            raise ValueError(f"{context}: input_scale must be non-zero.")
        return self

    def to_dict(self) -> ConfigDict:
        return {"method": "custom_range_scaler", "input_scale": float(self.input_scale), "centering": self.centering}


@dataclass(frozen=True)
class MinMaxScalerConfig(PreprocessingConfig):
    """Step 2 parameters for Min-Max Scaler (feature range scaling).
    Scales data to [feature_min, feature_max].
    """
    feature_min: float
    feature_max: float

    def validate(self, context: str = "min_max_scaler") -> MinMaxScalerConfig:
        if float(self.feature_max) <= float(self.feature_min):
            raise ValueError(f"{context}: feature_max must be greater than feature_min.")
        return self

    def to_dict(self) -> ConfigDict:
        return {
            "method": "min_max_scaler",
            "feature_min": float(self.feature_min),
            "feature_max": float(self.feature_max),
        }

@dataclass(frozen=True)
class AffineScalerConfig(PreprocessingConfig):
    """Step 2 parameters for Affine Scaler (y = X * scale + shift)."""
    input_scale: float
    shift: float

    def validate(self, context: str = "affine_scaler") -> AffineScalerConfig:
        _ = context
        return self

    def to_dict(self) -> ConfigDict:
        return {"method": "affine_scaler", "scale": float(self.input_scale), "shift": float(self.shift)}


@dataclass(frozen=True)
class RandomProjectionConfig(ProjectionConfig):
    """Step 3 parameters for Random Projection."""
    n_units: int
    input_scale: float
    input_connectivity: float
    bias_scale: float
    seed: int
    
    def validate(self, context: str = "random_projection") -> RandomProjectionConfig:
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

    def to_dict(self) -> ConfigDict:
        return {
            "type": "random",
            "n_units": int(self.n_units),
            "input_scale": float(self.input_scale),
            "input_connectivity": float(self.input_connectivity),
            "bias_scale": float(self.bias_scale),
            "seed": int(self.seed),
        }

@dataclass(frozen=True)
class CenterCropProjectionConfig(ProjectionConfig):
    """Step 3 parameters for Center Crop Projection (3D input only)."""
    n_units: int
    
    def validate(self, context: str = "center_crop") -> CenterCropProjectionConfig:
        prefix = f"{context}: "
        if int(self.n_units) <= 0:
            raise ValueError(f"{prefix}n_units must be positive.")
        return self

    def to_dict(self) -> ConfigDict:
        return {
            "type": "center_crop",
            "n_units": int(self.n_units),
        }


@dataclass(frozen=True)
class ResizeProjectionConfig(ProjectionConfig):
    """Step 3 parameters for Resize (Interpolation) Projection."""
    n_units: int

    def validate(self, context: str = "resize_projection") -> ResizeProjectionConfig:
        prefix = f"{context}: "
        if int(self.n_units) <= 0:
            raise ValueError(f"{prefix}n_units must be positive.")
        return self

    def to_dict(self) -> ConfigDict:
        return {
            "type": "resize",
            "n_units": int(self.n_units),
        }

@dataclass(frozen=True)
class PolynomialProjectionConfig(ProjectionConfig):
    """Step 3 parameters for Polynomial Projection (feature expansion)."""
    degree: int
    include_bias: bool

    def validate(self, context: str = "polynomial_projection") -> PolynomialProjectionConfig:
        if self.degree is None or int(self.degree) < 1:
            raise ValueError(f"{context}: degree must be >=1.")
        return self

    def to_dict(self) -> ConfigDict:
        return {"type": "polynomial", "degree": int(self.degree), "include_bias": self.include_bias}


@dataclass(frozen=True)
class PCAProjectionConfig(ProjectionConfig):
    """Step 3 parameters for PCA Projection (dimensionality reduction).
    
    Note: PCA assumes StandardScaler preprocessing (zero mean, unit variance).
    
    Args:
        n_units: Number of principal components to keep.
        input_scaler: Scalar to multiply output after projection.
    """
    n_units: int
    input_scaler: float

    def validate(self, context: str = "pca_projection") -> PCAProjectionConfig:
        if self.n_units is None or int(self.n_units) < 1:
            raise ValueError(f"{context}: n_units must be >=1.")
        return self

    def to_dict(self) -> ConfigDict:
        return {"type": "pca", "n_units": int(self.n_units), "input_scaler": float(self.input_scaler)}


@dataclass(frozen=True)
class ClassicalReservoirConfig(ModelConfig):
    """Step 5 and 6 reservoir dynamics parameters."""

    spectral_radius: float
    leak_rate: float
    rc_connectivity: float
    seed: int
    aggregation: AggregationMode

    def validate(self, context: str = "dynamics") -> ClassicalReservoirConfig:
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

    def to_dict(self) -> ConfigDict:
        return {
            "spectral_radius": float(self.spectral_radius),
            "leak_rate": float(self.leak_rate),
            "rc_connectivity": float(self.rc_connectivity),
            "seed": int(self.seed),
            "aggregation": self.aggregation.value,
        }

@dataclass(frozen=True)
class DistillationConfig(ModelConfig):
    """Configuration for distilling reservoir dynamics into a Student FNN."""
    """Step 5 and 6 distillation fnn dynamics parameters."""

    teacher: ClassicalReservoirConfig
    student: FNNConfig

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> ConfigDict:
        from typing import cast
        return cast("ConfigDict", {
            "teacher": self.teacher.to_dict(),
            "student.hidden_layers": tuple(int(v) for v in (self.student.hidden_layers or ())),
        })

    def validate(self, context: str = "") -> DistillationConfig:
        prefix = f"{context}: " if context else ""
        self.teacher.validate(context=f"{prefix}teacher")
        if not self.student.hidden_layers:
            raise ValueError(f"{prefix}student.hidden_layers must contain at least one layer size.")
        if any(width < 0 for width in self.student.hidden_layers):
            raise ValueError(f"{prefix}student.hidden_layers values must be non negative.")
        return self


@dataclass(frozen=True)
class FNNConfig(ModelConfig):
    """FNN configuration with optional sliding window for time series."""
    hidden_layers: tuple[int, ...] | None
    window_size: int | None = None  # None = Flatten, int = TimeDelayEmbedding(K)

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> ConfigDict:
        result: ConfigDict = {
            "hidden_layers": tuple(int(v) for v in (self.hidden_layers or ())),
        }
        if self.window_size is not None:
            result["window_size"] = int(self.window_size)
        return result

    def validate(self, context: str = "") -> FNNConfig:
        prefix = f"{context}: " if context else ""
        layers = self.hidden_layers or ()
        if any(width < 0 for width in layers):
            raise ValueError(f"{prefix}hidden_layers values must be non-negative.")
        if self.window_size is not None and int(self.window_size) < 1:
            raise ValueError(f"{prefix}window_size must be >= 1.")
        return self







@dataclass(frozen=True)
class PassthroughConfig(ModelConfig):
    """Configuration for passthrough model that skips dynamics (Step 5)."""
    aggregation: AggregationMode

    def validate(self, context: str = "passthrough") -> PassthroughConfig:
        if not isinstance(self.aggregation, AggregationMode):
            raise TypeError(f"{context}: aggregation must be AggregationMode, got {type(self.aggregation)}.")
        return self

    def to_dict(self) -> ConfigDict:
        return {
            "aggregation": self.aggregation.value,
        }


@dataclass(frozen=True)
class QuantumReservoirConfig(ModelConfig):
    """Step 5 Quantum Reservoir dynamics parameters.
    
    n_qubits: Optional. If None, inferred from projected_input_dim (Step 3 output).
              If specified, used directly (needed when projection=None).
    
    measurement_basis options:
        - 'Z': 1st moment only (n_qubits features)
        - 'ZZ': 2-point correlations only (n_qubits*(n_qubits-1)/2 features)
        - 'Z+ZZ': 1st moment + 2-point correlations (n_qubits + n_qubits*(n_qubits-1)/2 features)
                  For 4 qubits: 4 + 6 = 10 features
    """
    
    n_layers: int                    # Number of variational layers
    seed: int                        # Random seed for fixed parameters
    aggregation: AggregationMode     # How to aggregate time steps
    feedback_scale: float     # a_fb: R gate feedback scaling. 0.0 = no feedback
    measurement_basis: Literal["Z", "ZZ", "Z+ZZ"]
    noise_type: Literal["clean", "depolarizing", "damping"]
    noise_prob: float          # Probability of noise (0.0 to 1.0)
    readout_error: float     # Readout error probability (0.0 to 1.0)
    n_trajectories: int      # Number of trajectories for Monte Carlo simulation (0 = Density Matrix)
    use_remat: bool          # Use gradient checkpointing (rematerialization)
    use_reuploading: bool    # Use data re-uploading strategy
    precision: Literal["complex64", "complex128"]
    n_qubits: int | None = None   # Number of qubits (None = infer from Step 3)

    def validate(self, context: str = "quantum_reservoir") -> QuantumReservoirConfig:
        prefix = f"{context}: "
        if int(self.n_layers) <= 0:
            raise ValueError(f"{prefix}n_layers must be positive.")
        if self.n_qubits is not None and int(self.n_qubits) <= 0:
            raise ValueError(f"{prefix}n_qubits must be positive when specified.")
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

    def to_dict(self) -> ConfigDict:
        return {
            "n_qubits": int(self.n_qubits) if self.n_qubits is not None else None,
            "n_layers": int(self.n_layers),
            "seed": int(self.seed),
            "feedback_scale": float(self.feedback_scale),
            "aggregation": self.aggregation.value,
            "measurement_basis": str(self.measurement_basis),
            "noise_type": str(self.noise_type),
            "noise_prob": float(self.noise_prob),
            "readout_error": float(self.readout_error),
            "n_trajectories": int(self.n_trajectories),
            "use_remat": bool(self.use_remat),
            "use_reuploading": bool(self.use_reuploading),
            "precision": str(self.precision),
        }


@dataclass(frozen=True)
class RidgeReadoutConfig(ReadoutConfig):
    """Step 7 readout configuration (structure/defaults)."""
    use_intercept: bool
    lambda_candidates: tuple[float, ...] | None = None

    def validate(self, context: str = "ridgereadout") -> RidgeReadoutConfig:
        if self.lambda_candidates is not None:
            if any(float(lam) <= 0.0 for lam in self.lambda_candidates):
                raise ValueError(f"{context}: lambda_candidates must contain only positive values.")
        return self

    def to_dict(self) -> ConfigDict:
        result: ConfigDict = {"use_intercept": bool(self.use_intercept)}
        if self.lambda_candidates is not None:
            result["lambda_candidates"] = [float(v) for v in self.lambda_candidates]
        return result



@dataclass(frozen=True)
class PolyRidgeReadoutConfig(ReadoutConfig):
    """Step 7 poly readout configuration (structure/defaults)."""

    use_intercept: bool
    lambda_candidates: tuple[float, ...] | None
    degree: int
    mode: Literal["full", "square_only"]

    def validate(self, context: str = "polyridgereadout") -> PolyRidgeReadoutConfig:
        if self.lambda_candidates is not None:
            if any(float(lam) <= 0.0 for lam in self.lambda_candidates):
                raise ValueError(f"{context}: lambda_candidates must contain only positive values.")
        if int(self.degree) < 2:
            raise ValueError(f"{context}: degree must be >= 2.")
        if self.mode not in ("full", "square_only"):
            raise ValueError(f"{context}: mode must be 'full' or 'square_only', got '{self.mode}'.")
        return self

    def to_dict(self) -> ConfigDict:
        result: ConfigDict = {"use_intercept": bool(self.use_intercept), "degree": int(self.degree), "mode": str(self.mode)}
        if self.lambda_candidates is not None:
            result["lambda_candidates"] = [float(v) for v in self.lambda_candidates]
        return result


@dataclass(frozen=True)
class FNNReadoutConfig(ReadoutConfig):
    """Step 7 readout configuration (structure/fnn)."""
    hidden_layers: tuple[int, ...] | None

    def validate(self, context: str = "fnnreadout") -> FNNReadoutConfig:
        _ = context
        return self

    def to_dict(self) -> ConfigDict:
        return {"hidden_layers": tuple(self.hidden_layers or ())}



