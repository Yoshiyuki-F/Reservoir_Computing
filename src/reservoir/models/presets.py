"""src/reservoir/models/presets.py
Central Registry for Model Presets.
Aggregates configurations from sub-modules into named presets for the CLI.
Uses shared identifiers from reservoir.core.identifiers.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Union

from reservoir.core.presets import PresetRegistry
from reservoir.core.identifiers import Pipeline
from reservoir.models.reservoir.classical.config import ClassicalReservoirConfig
from reservoir.models.distillation.config import DistillationConfig

# -----------------------------------------------------------------------------
# Type Definitions
# -----------------------------------------------------------------------------

ModelConfiguration = Union[ClassicalReservoirConfig, DistillationConfig]


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """
    Wrapper for model configuration.
    Links a canonical pipeline ID (model_type) with a specific config object.
    """

    name: str
    model_type: Pipeline
    description: str = ""
    config: Optional[ModelConfiguration] = None
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def reservoir(self) -> Optional[ClassicalReservoirConfig]:
        """Expose reservoir configs for consumers expecting the legacy attribute."""
        if isinstance(self.config, ClassicalReservoirConfig):
            return self.config
        return None

    @property
    def distillation(self) -> Optional[DistillationConfig]:
        """Expose distillation configs for consumers expecting the legacy attribute."""
        if isinstance(self.config, DistillationConfig):
            return self.config
        return None

    def to_params(self) -> Dict[str, Any]:
        """Merge explicit params with the underlying config dict."""
        merged: Dict[str, Any] = {}

        if self.config is not None:
            if isinstance(self.config, DistillationConfig):
                merged.update(self.config.teacher.to_dict())
                merged["student_hidden_layers"] = tuple(int(v) for v in self.config.student_hidden_layers)
            elif hasattr(self.config, "to_dict"):
                merged.update(self.config.to_dict())
            else:
                merged.update(asdict(self.config))

        merged.update(self.params)
        return merged

    def __getattr__(self, item: str) -> Any:
        """
        Provide passthrough access to underlying config/params for
        compatibility (e.g., preset.leak_rate in tests).
        """
        if self.config is not None and hasattr(self.config, item):
            return getattr(self.config, item)
        if isinstance(self.config, DistillationConfig) and hasattr(self.config.teacher, item):
            return getattr(self.config.teacher, item)
        if item in self.params:
            return self.params[item]
        raise AttributeError(f"{item} not found in ModelConfig or underlying config.")


# -----------------------------------------------------------------------------
# Definitions
# -----------------------------------------------------------------------------

MODEL_DEFINITIONS: Dict[str, ModelConfig] = {
    Pipeline.CLASSICAL_RESERVOIR.value: ModelConfig(
        name="classical",
        model_type=Pipeline.CLASSICAL_RESERVOIR,
        description="Standard classical reservoir (Echo State Network)",
        config=ClassicalReservoirConfig(),
    ),
    Pipeline.FNN_DISTILLATION.value: ModelConfig(
        name="fnn-distillation",
        model_type=Pipeline.FNN_DISTILLATION,
        description="Feedforward Neural Network with Reservoir Distillation",
        config=DistillationConfig(),
    ),
}


# -----------------------------------------------------------------------------
# Aliases
# -----------------------------------------------------------------------------

MODEL_ALIASES: Dict[str, str] = {
    # Classical
    "classical": Pipeline.CLASSICAL_RESERVOIR.value,
    "reservoir": Pipeline.CLASSICAL_RESERVOIR.value,
    "esn": Pipeline.CLASSICAL_RESERVOIR.value,
    "classical_reservoir": Pipeline.CLASSICAL_RESERVOIR.value,
    # FNN / Distillation
    "fnn": Pipeline.FNN_DISTILLATION.value,
    "distillation": Pipeline.FNN_DISTILLATION.value,
    "fnn_distillation": Pipeline.FNN_DISTILLATION.value,
    "fnn-distillation": Pipeline.FNN_DISTILLATION.value,
}


# -----------------------------------------------------------------------------
# Registry Setup
# -----------------------------------------------------------------------------

MODEL_REGISTRY = PresetRegistry(MODEL_DEFINITIONS, aliases=MODEL_ALIASES)

# Flattened dictionary for compatibility (direct .get access without registry)
MODEL_PRESETS: Dict[str, ModelConfig] = {}
MODEL_PRESETS.update(MODEL_DEFINITIONS)
for alias, target_key in MODEL_ALIASES.items():
    if target_key in MODEL_DEFINITIONS:
        MODEL_PRESETS[alias] = MODEL_DEFINITIONS[target_key]


def get_model_preset(name: str) -> ModelConfig:
    """Retrieves a model preset by name or alias."""
    return MODEL_REGISTRY.get_or_default(name, default_key=Pipeline.CLASSICAL_RESERVOIR.value)


__all__ = [
    "ModelConfig",
    "ModelConfiguration",
    "MODEL_DEFINITIONS",
    "MODEL_ALIASES",
    "MODEL_REGISTRY",
    "MODEL_PRESETS",
    "get_model_preset",
]
