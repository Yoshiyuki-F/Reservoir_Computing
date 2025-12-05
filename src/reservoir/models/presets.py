"""src/reservoir/models/presets.py
Central Registry for Model Presets.
Aggregates configurations from sub-modules into named presets for the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from reservoir.core.presets import PresetRegistry

# ★ Import Configs from their new homes
from reservoir.models.reservoir.config import ReservoirConfig
from reservoir.models.distillation.config import DistillationConfig


@dataclass(frozen=True)
class ModelPreset:
    """Wrapper for a specific combination of model config and default parameters."""
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


# Define Standard Presets
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
            # This will be hydrated into DistillationConfig by the Factory
            "distillation": DistillationConfig(),
        },
    ),
}

MODEL_ALIASES: Dict[str, str] = {
    "cr": "classical",
    "esn": "classical",
    "qrc": "quantum_gate_based",
    "qa": "quantum_analog",
    "distill": "distillation_bottleneck",
}

MODEL_REGISTRY = PresetRegistry(MODEL_DEFINITIONS, MODEL_ALIASES)
MODEL_PRESETS = MODEL_DEFINITIONS

def get_model_preset(name: str) -> ModelPreset:
    return MODEL_REGISTRY.get_or_default(name, "classical")

__all__ = [
    "ReservoirConfig",
    "DistillationConfig",
    "ModelPreset",
    "MODEL_PRESETS",
    "get_model_preset",
]
