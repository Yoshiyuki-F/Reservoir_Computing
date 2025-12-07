"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/presets.py
Central registry for model presets.
SSOT: all default hyperparameters live in these dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Union

from reservoir.core.presets import StrictRegistry
from reservoir.core.identifiers import AggregationMode, Preprocessing, Model
from reservoir.models.config import (
    PreprocessingConfig,
    ProjectionConfig,
    ClassicalReservoirConfig,
    DistillationConfig,
)


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
    projection: ProjectionConfig
    model: Union[ClassicalReservoirConfig, DistillationConfig]

    def __post_init__(self) -> None:
        if self.preprocess is None:
            raise ValueError(f"{self.name}: preprocess config is required.")
        if self.projection is None:
            raise ValueError(f"{self.name}: projection config is required.")
        if self.model is None:
            raise ValueError(f"{self.name}: model config is required.")

        self.preprocess.validate(context=f"{self.name}.preprocess")
        self.projection.validate(context=f"{self.name}.projection")

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
            merged["student_hidden_layers"] = tuple(int(v) for v in model_cfg.student_hidden_layers)
        elif hasattr(model_cfg, "to_dict"):
            merged.update(model_cfg.to_dict())
        else:
            merged.update(asdict(model_cfg))

        merged.update(self.preprocess.to_dict())
        merged.update(self.projection.to_dict())

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


# -----------------------------------------------------------------------------
# Definitions
# -----------------------------------------------------------------------------

DEFAULT_PREPROCESS = PreprocessingConfig(
    method=Preprocessing.RAW,
    poly_degree=1,
)

DEFAULT_PROJECTION = ProjectionConfig(
    n_units=100,
    input_scale=0.6,
    input_connectivity=0.1,
    bias_scale=1.0,
    seed=42,
)


CLASSICAL_RESERVOIR_DYNAMICS = ClassicalReservoirConfig(
    spectral_radius=1.3,
    leak_rate=0.2,
    rc_connectivity=0.9,
    seed=42,
    aggregation=AggregationMode.MEAN,
)

FNN_DISTILLATION_DYNAMICS = DistillationConfig(
    teacher=CLASSICAL_RESERVOIR_DYNAMICS,
    student_hidden_layers=(100,),
)


FNN_DISTILLATION_PRESET = PipelineConfig(
    name="fnn-distillation",
    model_type=Model.FNN_DISTILLATION,
    description="Feedforward Neural Network with Reservoir Distillation",
    preprocess=DEFAULT_PREPROCESS,
    projection=DEFAULT_PROJECTION,
    model=FNN_DISTILLATION_DYNAMICS,
)

CLASSICAL_RESERVOIR_PRESET = PipelineConfig(
    name="classical-reservoir",
    model_type=Model.CLASSICAL_RESERVOIR,
    description="Echo State Network (Classical Reservoir Computing)",
    preprocess=DEFAULT_PREPROCESS,
    projection=DEFAULT_PROJECTION,
    model=CLASSICAL_RESERVOIR_DYNAMICS,
)


MODEL_DEFINITIONS: Dict[Model, PipelineConfig] = {
    Model.CLASSICAL_RESERVOIR: CLASSICAL_RESERVOIR_PRESET,
    Model.FNN_DISTILLATION: FNN_DISTILLATION_PRESET,
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
