"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/presets.py
Central Registry for Model Presets.
Aggregates configurations from sub-modules into named presets for the CLI.
Uses shared identifiers from reservoir.core.identifiers.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Union

from reservoir.core.presets import StrictRegistry
from reservoir.core.identifiers import AggregationMode, Preprocessing,  Model
from reservoir.models.config import PreprocessingConfig, ProjectionConfig, AggregationConfig, ClassicalReservoirConfig, \
    DistillationConfig


@dataclass(frozen=True)
class PipelineConfig:
    """
    Wrapper for model configuration.
    Links a canonical pipeline ID (model_type) with a specific config object.
    """

    name: str
    model_type: Model
    description: str
    preprocess_config: PreprocessingConfig
    projection_config: ProjectionConfig
    model_config: Union[ClassicalReservoirConfig, DistillationConfig]
    aggregation_config: Optional[AGGREGATION_CONFIG]

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
        raise AttributeError(f"{item} not found in PipelineConfig or underlying config.")


# -----------------------------------------------------------------------------
# Definitions
# -----------------------------------------------------------------------------

PREPROCESSING_CONFIG = PreprocessingConfig(
    method=Preprocessing.RAW,
    poly_degree=1
)

PROJECTION_CONFIG = ProjectionConfig(
    n_units=100,
    input_scale=0.6,
    input_connectivity=0.1,
    bias_scale=1.0,
    seed=42
)

CLASSICAL_RESERVOIR_CONFIG = ClassicalReservoirConfig(
    spectral_radius=1.3,
    leak_rate=0.2,
    rc_connectivity=0.9,
    seed=42
)

DISTILLATION_CONFIG = DistillationConfig(
    teacher=CLASSICAL_RESERVOIR_CONFIG,
    student_hidden_layers=(100,),
)

AGGREGATION_CONFIG = AggregationConfig(
    mode=AggregationMode.MEAN
)


FNN_DISTILLATION_CONFIG = PipelineConfig(
    name="fnn-distillation",
    model_type=Model.FNN_DISTILLATION,
    description="Feedforward Neural Network with Reservoir Distillation",
    preprocess_config=PREPROCESSING_CONFIG,
    projection_config=PROJECTION_CONFIG,
    model_config=DISTILLATION_CONFIG,
    aggregation_config=None
)

CLASSICAL_RESERVOIR_CONFIG = PipelineConfig(
    name="classical-reservoir",
    model_type=Model.CLASSICAL_RESERVOIR,
    description="Echo State Network (Classical Reservoir Computing)",
    preprocess_config=PREPROCESSING_CONFIG,
    projection_config=PROJECTION_CONFIG,
    model_config=CLASSICAL_RESERVOIR_CONFIG,
    aggregation_config=AGGREGATION_CONFIG
)

MODEL_DEFINITIONS: Dict[Model, PipelineConfig] = {
    Model.CLASSICAL_RESERVOIR: CLASSICAL_RESERVOIR_CONFIG,
    Model.FNN_DISTILLATION: FNN_DISTILLATION_CONFIG,
}


# -----------------------------------------------------------------------------
# Registry Setup
# -----------------------------------------------------------------------------

MODEL_REGISTRY = StrictRegistry(MODEL_DEFINITIONS)

MODEL_PRESETS: Dict[Model, PipelineConfig] = dict(MODEL_DEFINITIONS)


def get_model_preset(model: Model) -> PipelineConfig:
    """Retrieves a model preset by enum key; raises on invalid names."""
    preset = MODEL_REGISTRY.get(model)
    if preset is None:
        raise KeyError(f"Model preset '{model}' not found.")
    return preset


__all__ = [
    "PipelineConfig",
    "MODEL_DEFINITIONS",
    "MODEL_REGISTRY",
    "MODEL_PRESETS",
    "get_model_preset",
]
