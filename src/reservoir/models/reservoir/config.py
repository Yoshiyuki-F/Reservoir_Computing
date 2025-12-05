"""src/reservoir/models/reservoir/config.py"""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ReservoirConfig:
    """
    Configuration for reservoir nodes (Classical/Quantum).
    SSOT: Defaults defined here are the canonical defaults.
    """

    n_units: Optional[int] = None  # defined in scripts/CLI
    spectral_radius: float = 1.3
    leak_rate: float = 0.2
    input_scale: float = 0.6
    input_connectivity: float = 0.9
    rc_connectivity: float = 0.1
    bias_scale: float = 1.0
    noise_rc: float = 0.001
    seed: int = 42
    use_design_matrix: bool = False
    poly_degree: int = 1
    state_aggregation: str = "mean"

    # Quantum / advanced parameters
    nonlinearity: Optional[str] = None
    encode_batch_size: Optional[int] = None
    coupling: Optional[float] = None
    dt: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def validate(self, *, context: str = "") -> None:
        prefix = f"{context}: " if context else ""
        if self.n_units is None or self.n_units <= 0:
            raise ValueError(f"{prefix}n_units must be > 0.")
        if not (0.0 < self.spectral_radius):
            raise ValueError(f"{prefix}spectral_radius must be > 0.")
        if not (0.0 <= self.leak_rate <= 1.0):
            raise ValueError(f"{prefix}leak_rate must be in [0, 1].")
        if not (0.0 < self.input_scale):
            raise ValueError(f"{prefix}input_scale must be > 0.")
        if self.bias_scale < 0.0:
            raise ValueError(f"{prefix}bias_scale must be >= 0.")
