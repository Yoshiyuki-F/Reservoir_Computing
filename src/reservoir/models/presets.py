"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/presets.py
Central registry for model presets.
SSOT: all default hyperparameters live in these dataclasses.
"""
from __future__ import annotations

from typing import Dict

from reservoir.core.presets import StrictRegistry
from reservoir.core.identifiers import AggregationMode, Preprocessing, Model
from reservoir.models.config import (
    PreprocessingConfig,
    ProjectionConfig,
    ClassicalReservoirConfig,
    DistillationConfig, FNNConfig, PipelineConfig, ReadoutConfig,
)
from reservoir.readout import RidgeRegression

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

DEFAULT_READOUT = ReadoutConfig(
    model = RidgeRegression(10000, use_intercept=True)
)

CLASSICAL_RESERVOIR_DYNAMICS = ClassicalReservoirConfig(
    spectral_radius=1.3,
    leak_rate=0.2,
    rc_connectivity=0.9,
    seed=42,
    aggregation=AggregationMode.MEAN,
)

FNN_DYNAMICS = FNNConfig(
    hidden_layers=(100,),
)

FNN_DISTILLATION_PRESET = PipelineConfig(
    name="fnn-distillation",
    model_type=Model.FNN_DISTILLATION,
    description="Feedforward Neural Network with Reservoir Distillation",
    preprocess=DEFAULT_PREPROCESS,
    projection=DEFAULT_PROJECTION,
    readout=DEFAULT_READOUT,
    model=DistillationConfig(
        teacher=CLASSICAL_RESERVOIR_DYNAMICS,
        student=FNN_DYNAMICS,
    ),
)

CLASSICAL_RESERVOIR_PRESET = PipelineConfig(
    name="classical-reservoir",
    model_type=Model.CLASSICAL_RESERVOIR,
    description="Echo State Network (Classical Reservoir Computing)",
    preprocess=DEFAULT_PREPROCESS,
    projection=DEFAULT_PROJECTION,
    model=CLASSICAL_RESERVOIR_DYNAMICS,
    readout=DEFAULT_READOUT
)

FNN_PRESET = PipelineConfig(
    name="fnn",
    model_type=Model.FNN,
    description="Feedforward Neural Network (FNN)",
    preprocess=DEFAULT_PREPROCESS,
    projection=None,
    model=FNN_DYNAMICS,
    readout=None
)

MODEL_DEFINITIONS: Dict[Model, PipelineConfig] = {
    Model.CLASSICAL_RESERVOIR: CLASSICAL_RESERVOIR_PRESET,
    Model.FNN_DISTILLATION: FNN_DISTILLATION_PRESET,
    Model.FNN: FNN_PRESET
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
