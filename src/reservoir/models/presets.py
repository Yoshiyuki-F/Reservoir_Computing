# /home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/presets.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from reservoir.core.presets import PresetRegistry


@dataclass(frozen=True)
class ReservoirConfig:
    """
    Configuration for reservoir nodes.
    SSOT: Defaults defined here are the canonical defaults.
    """

    n_units: Optional[int] = None # should be defined in scripts
    spectral_radius: float = 1.3
    leak_rate: float = 0.2
    input_scale: float = 0.6
    input_connectivity: float = 0.9 # 0.09 # optimised for rs100? sus
    rc_connectivity: float = 0.1  #0.59 # optimised for rs100
    bias_scale: float = 1.0
    noise_rc: float = 0.001
    seed: int = 42
    use_design_matrix: bool = False
    poly_degree: int = 1
    state_aggregation: str = "mean"

    # Quantum / advanced
    nonlinearity: Optional[str] = None
    encode_batch_size: Optional[int] = None
    coupling: Optional[float] = None
    dt: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def validate(self, *, context: str = "") -> None:
        prefix = f"{context}: " if context else ""
        if self.n_units is None or self.n_units <= 0:
            raise ValueError(f"{prefix}n_units must be explicitly defined and > 0 for reservoir operation.")
        if not (0.0 < self.spectral_radius):
            raise ValueError(f"{prefix}spectral_radius must be > 0.")
        if not (0.0 <= self.leak_rate <= 1.0):
            raise ValueError(f"{prefix}leak_rate must be in [0, 1].")
        if not (0.0 < self.input_scale):
            raise ValueError(f"{prefix}input_scale must be > 0.")
        if not (0.0 < self.input_connectivity <= 1.0):
            raise ValueError(f"{prefix}input_connectivity must be in (0, 1].")
        if not (0.0 < self.rc_connectivity <= 1.0):
            raise ValueError(f"{prefix}rc_connectivity must be in (0, 1].")
        if self.bias_scale < 0.0:
            raise ValueError(f"{prefix}bias_scale must be >= 0.")
        if self.noise_rc < 0.0:
            raise ValueError(f"{prefix}noise_rc must be >= 0.")


@dataclass(frozen=True)
class DistillationConfig:
    """Configuration for distilling reservoir dynamics into a feed-forward student."""

    teacher: ReservoirConfig = field(default_factory=ReservoirConfig)
    student_hidden_layers: Tuple[int, ...] = (10,)
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32

    def __post_init__(self) -> None:
        if not self.student_hidden_layers:
            raise ValueError("DistillationConfig.student_hidden_layers must contain at least one layer size.")
        if any(width <= 0 for width in self.student_hidden_layers):
            raise ValueError("DistillationConfig.student_hidden_layers values must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("DistillationConfig.learning_rate must be positive.")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("DistillationConfig.epochs and batch_size must be positive.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "teacher": self.teacher.to_dict(),
            "student_hidden_layers": tuple(int(v) for v in self.student_hidden_layers),
            "learning_rate": float(self.learning_rate),
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
        }

    def validate(self, *, context: str = "") -> None:
        prefix = f"{context}: " if context else ""
        self.teacher.validate(context=f"{prefix}teacher")
        if not self.student_hidden_layers:
            raise ValueError(f"{prefix}student_hidden_layers must contain at least one layer size.")
        if any(width <= 0 for width in self.student_hidden_layers):
            raise ValueError(f"{prefix}student_hidden_layers values must be positive.")
        if self.learning_rate <= 0:
            raise ValueError(f"{prefix}learning_rate must be positive.")
        if self.epochs <= 0:
            raise ValueError(f"{prefix}epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError(f"{prefix}batch_size must be positive.")


@dataclass(frozen=True)
class ModelPreset:
    name: str
    model_type: str
    description: str = ""
    reservoir: Optional[ReservoirConfig] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def to_params(self) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        if self.reservoir is not None:
            merged.update(self.reservoir.to_dict())
        merged.update(self.params)
        return merged


MODEL_DEFINITIONS: Dict[str, ModelPreset] = {
    "classical": ModelPreset(
        name="classical",
        model_type="reservoir",
        description="Standard classical reservoir (Echo State Network)",
        reservoir=ReservoirConfig()
    ),
    "quantum_gate_based": ModelPreset(
        name="quantum_gate_based",
        model_type="quantum_gate_based",
        description="Gate-based Quantum Reservoir Computer",
        reservoir=ReservoirConfig(),
        params={
            "n_qubits": 4,
            "circuit_depth": 4,
            "backend": "default.qubit",
            "entanglement": "full",
            "encoding_scheme": "detuning",
            "state_aggregation": "last",
        },
    ),
    "quantum_analog": ModelPreset(
        name="quantum_analog",
        model_type="quantum_analog",
        description="Analog Quantum Reservoir parameters",
        reservoir=ReservoirConfig(dt=0.1),
        params={
            "n_qubits": 6,
            "encoding_scheme": "detuning",
            "state_aggregation": "last_mean",
        },
    ),
    "distillation_bottleneck": ModelPreset(
        name="distillation_bottleneck",
        model_type="distillation",
        description="Distill reservoir trajectories into a compact FNN bottleneck.",
        params={
            "distillation": DistillationConfig(),
        },
    ),
}

MODEL_ALIASES: Dict[str, str] = {
    "cr": "classical",
    "esn": "classical",
    "qrc": "quantum_gate_based",
    "qa": "quantum_analog",
    "gatebased_quantum": "quantum_gate_based",
    "analog_quantum": "quantum_analog",
}

MODEL_REGISTRY = PresetRegistry(MODEL_DEFINITIONS, MODEL_ALIASES)
MODEL_PRESETS = MODEL_DEFINITIONS


def normalize_model_name(name: str) -> str:
    return MODEL_REGISTRY.normalize_name(name)


def get_model_preset(name: str) -> ModelPreset:
    return MODEL_REGISTRY.get_or_default(name, "classical")


__all__ = [
    "ReservoirConfig",
    "DistillationConfig",
    "ModelPreset",
    "MODEL_PRESETS",
    "MODEL_REGISTRY",
    "MODEL_ALIASES",
    "get_model_preset",
    "normalize_model_name",
]
