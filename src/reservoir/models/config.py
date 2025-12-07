from dataclasses import dataclass
from reservoir.core.identifiers import AggregationMode, Preprocessing, ReadOutType


# --- Step 2: Preprocessing Config ---
@dataclass(frozen=True)
class PreprocessingConfig:
    method: Preprocessing
    poly_degree: int

# --- Step 3: Projection Config (The Bridge) ---
@dataclass(frozen=True)
class ProjectionConfig:
    n_units: int
    input_scale: float
    input_connectivity: float
    bias_scale: float
    seed: int

    def validate(self, *, context: str = "") -> None:
        """Validates physical constraints across all components."""
        prefix = f"{context}: " if context else ""

        # Projection Checks
        if self.n_units <= 0:
            raise ValueError(f"{prefix}n_units must be > 0.")
        if self.input_scale < 0:
            raise ValueError(f"{prefix}input_scale must be >= 0.")
        if not (0.0 <= self.input_connectivity <= 1.0):
            raise ValueError(f"{prefix}input_connectivity must be in [0, 1].")
        if self.bias_scale < 0:
            raise ValueError(f"{prefix}bias_scale must be >= 0.")


# --- Step 5: Dynamics Config (The Engine) ---
@dataclass(frozen=True)
class ReservoirDynamicsConfig:
    spectral_radius: float
    leak_rate: float
    rc_connectivity: float
    seed: int

    def validate(self, *, context: str = "") -> None:
        """Validates physical constraints across all components."""
        prefix = f"{context}: " if context else ""

        # Dynamics Checks
        if self.spectral_radius <= 0:
            raise ValueError(f"{prefix}spectral_radius must be > 0.")
        if not (0.0 <= self.leak_rate <= 1.0):
            raise ValueError(f"{prefix}leak_rate must be in [0, 1].")


# --- Step 6: Aggregation Config ---
@dataclass(frozen=True)
class AggregationConfig:
    mode: AggregationMode

    def validate(self, *, context: str = "") -> None:
        """Validates physical constraints across all components."""
        prefix = f"{context}: " if context else ""

        # Aggregation Checks
        if not isinstance(self.mode, AggregationMode):
            raise ValueError(f"{prefix}mode must be a valid AggregationMode enum.")

# --- Step 7: Readout Config (not used in ClassicalReservoirConfig) ---
@dataclass(frozen=True)
class ReadoutConfig:
    mode: ReadOutType