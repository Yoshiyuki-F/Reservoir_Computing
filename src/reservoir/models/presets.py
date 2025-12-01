from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from reservoir.core.presets import PresetRegistry


@dataclass(frozen=True)
class ReservoirConfig:
    """Configuration for reservoir nodes (Classical & Quantum)."""

    # --- Core Physics Parameters (Canonical Names) ---
    n_units: int = None           # Removed it should be defined in script
    spectral_radius: float = 1.3
    leak_rate: float = 0.2        # Removed 'alpha'
    input_scale: float = 0.6      # Removed 'input_scaling'
    connectivity: float = 0.1     # Removed 'sparsity'
    bias_scale: float = 1.0       # Removed 'input_bias'
    noise_rc: float = 0.001       # Removed 'noise_level'
    seed: int = 42                # Removed 'random_seed'
    use_design_matrix: bool = False
    poly_degree: int = 2
    state_aggregation: str = "mean"

    # --- Advanced / Quantum Specifics (Optional) ---
    # These are distinct parameters, not aliases.
    nonlinearity: Optional[str] = None      # e.g., 'tanh', 'relu'
    encode_batch_size: Optional[int] = None # For quantum batch processing
    coupling: Optional[float] = None        # Interaction strength (Quantum)
    dt: Optional[float] = None              # Time step (Quantum/ODE)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, filtering out None values."""
        # Note: We do NOT generate aliases here.
        # Consumers must use canonical names (e.g., 'leak_rate', not 'alpha').
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None
        }


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


# --- Preset Definitions ---

MODEL_DEFINITIONS: Dict[str, ModelPreset] = {
    "classical": ModelPreset(
        name="classical",
        model_type="reservoir",
        description="Standard classical reservoir (Echo State Network)",
        # Default values come from ReservoirConfig class definition.
        # Only override what differs from the defaults.
        reservoir=ReservoirConfig(noise_rc=0.0),
    ),
    "quantum_gate_based": ModelPreset(
        name="quantum_gate_based",
        model_type="quantum_gate_based",
        description="Gate-based Quantum Reservoir Computer",
        reservoir=ReservoirConfig(n_units=4),
        params={
            # Quantum-specific structure parameters
            "n_qubits": 4,
            "circuit_depth": 4,
            "backend": "default.qubit",
            "entanglement": "full",
            "encoding_scheme": "detuning",
        },
    ),
    "quantum_analog": ModelPreset(
        name="quantum_analog",
        model_type="quantum_analog",
        description="Analog Quantum Reservoir parameters",
        reservoir=ReservoirConfig(
            n_units=6,
            dt=0.1, # Specific to analog dynamics
        ),
        params={
            "n_qubits": 6,
            "encoding_scheme": "detuning",
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
    "ModelPreset",
    "MODEL_PRESETS",
    "MODEL_REGISTRY",
    "MODEL_ALIASES",
    "get_model_preset",
    "normalize_model_name",
]