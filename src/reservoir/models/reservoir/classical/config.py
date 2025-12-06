"""src/reservoir/models/reservoir/classical/config.py"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

from reservoir.core.identifiers import AggregationMode


@dataclass(frozen=True)
class ClassicalReservoirConfig:
    """
    Configuration for reservoir nodes (Classical/Quantum).
    Strict Schema: physical hyperparameters are required; no implicit defaults.
    """

    n_units: int
    spectral_radius: float
    leak_rate: float
    input_scale: float
    input_connectivity: float
    rc_connectivity: float
    bias_scale: float
    noise_rc: float = 0.0
    seed: Optional[int] = None
    use_design_matrix: bool = False
    poly_degree: int = 1
    state_aggregation: AggregationMode = AggregationMode.MEAN

    def __post_init__(self) -> None:
        if isinstance(self.state_aggregation, str):
            try:
                object.__setattr__(self, "state_aggregation", AggregationMode(self.state_aggregation))
            except Exception as exc:
                raise ValueError(f"Invalid state_aggregation '{self.state_aggregation}'") from exc

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        mode = data.get("state_aggregation")
        if isinstance(mode, AggregationMode):
            data["state_aggregation"] = mode.value
        return data

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
        if not isinstance(self.state_aggregation, AggregationMode):
            raise TypeError(f"{prefix}state_aggregation must be AggregationMode, got {type(self.state_aggregation)}.")
