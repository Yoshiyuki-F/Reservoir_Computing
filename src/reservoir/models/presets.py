"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/presets.py
Central registry for model presets.
SSOT: all default hyperparameters live in these dataclasses.
"""
from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from reservoir.core.presets import StrictRegistry
from reservoir.core.identifiers import AggregationMode, Preprocessing, Model, Dataset
from reservoir.models.config import (
    PreprocessingConfig,
    ProjectionConfig,
    ClassicalReservoirConfig,
    DistillationConfig,
    FNNConfig,
    PipelineConfig,
    RidgeReadoutConfig, FNNReadoutConfig, PassthroughConfig
)
from reservoir.data.presets import get_dataset_preset 

def get_model_preset(model: Model, dataset: Dataset) -> PipelineConfig:
    """Retrieves a model preset by enum key; raises on invalid names."""
    # Check if regression task to use time-series optimized preset
    ds_preset = get_dataset_preset(dataset)

    if not ds_preset.classification and model == Model.CLASSICAL_RESERVOIR:
        return TIME_CLASSICAL_RESERVOIR_PRESET
    
    preset = StrictRegistry(MODEL_PRESETS).get(model)
    if preset is None:
        raise KeyError(f"Model preset '{model}' not found.")
    return preset

# -----------------------------------------------------------------------------
# Definitions
# -----------------------------------------------------------------------------

DEFAULT_PREPROCESS = PreprocessingConfig(
    method=Preprocessing.MAX_SCALER,
    poly_degree=1,
)

DEFAULT_PROJECTION = ProjectionConfig(
    n_units=100,
    input_scale=0.6,
    input_connectivity=0.1,
    bias_scale=1.0,
    seed=1,
)

DEFAULT_RIDGE_READOUT = RidgeReadoutConfig(
    init_lambda=1e-3,
    use_intercept=True,
    lambda_candidates=tuple(np.logspace(-12, 3, 30).tolist())
)

DEFAULT_FNN_READOUT = FNNReadoutConfig(hidden_layers=(1000,))


"=============================================RESERVOIR and Distillation Classification Presets============================================"

CLASSICAL_RESERVOIR_DYNAMICS = ClassicalReservoirConfig(
    spectral_radius=1.3,
    leak_rate=0.2,
    rc_connectivity=0.9,
    seed=42,
    aggregation=AggregationMode.MEAN,
)

CLASSICAL_RESERVOIR_PRESET = PipelineConfig(
    name="classical_reservoir",
    model_type=Model.CLASSICAL_RESERVOIR,
    description="Echo State Network (Classical Reservoir Computing)",
    preprocess=DEFAULT_PREPROCESS,
    projection=DEFAULT_PROJECTION,
    model=CLASSICAL_RESERVOIR_DYNAMICS,
    readout=DEFAULT_FNN_READOUT
)

FNN_DISTILLATION_PRESET = PipelineConfig(
    name="fnn-distillation",
    model_type=Model.FNN_DISTILLATION,
    description="Feedforward Neural Network with Reservoir Distillation",
    preprocess=DEFAULT_PREPROCESS,
    projection=DEFAULT_PROJECTION,
    model=DistillationConfig(
        teacher=CLASSICAL_RESERVOIR_DYNAMICS,
        student=FNNConfig(
            hidden_layers=(1000,1000),
        ),
    ),
    readout=DEFAULT_RIDGE_READOUT
)

"=============================================Time series Presets============================================"

DEFAULT_PROJECTION_REGRESSION = ProjectionConfig(
    n_units=1000,
    input_scale=0.2,
    input_connectivity=1.0,
    bias_scale=0.1,
    seed=1,
)

TIME_CLASSICAL_RESERVOIR_PRESET = PipelineConfig(
    name="classical_reservoir",
    model_type=Model.CLASSICAL_RESERVOIR,
    description="Echo State Network (Classical Reservoir Computing)",
    preprocess=PreprocessingConfig(
        method=Preprocessing.STANDARD_SCALER,
        poly_degree=1,
    ),
    projection=DEFAULT_PROJECTION_REGRESSION,
    model=ClassicalReservoirConfig(
        spectral_radius=1,
        leak_rate=0.4, #best for n ,not for n=1000
        rc_connectivity=0.02, #best
        seed=42,
        aggregation=AggregationMode.SEQUENCE,
    ),
    readout=DEFAULT_RIDGE_READOUT
)

PASSTHROUGH_PRESET = PipelineConfig(
    name="passthrough",
    model_type=Model.PASSTHROUGH,
    description="Passthrough model (Projection -> Aggregation, no dynamics)",
    preprocess=PreprocessingConfig(
        method=Preprocessing.STANDARD_SCALER,
        poly_degree=1,
    ),
    projection=DEFAULT_PROJECTION_REGRESSION,
    model=PassthroughConfig(
        aggregation=AggregationMode.SEQUENCE,
    ),
    readout=DEFAULT_RIDGE_READOUT
)


"=============================================FNN EndToEnd Presets============================================"

FNN_PRESET = PipelineConfig(
    name="fnn",
    model_type=Model.FNN,
    description="Feedforward Neural Network (FNN)",
    preprocess=PreprocessingConfig(
        method=Preprocessing.MAX_SCALER,
        poly_degree=1,
    ),
    projection=None,
    model=FNNConfig(
        hidden_layers=(100,),
    ),
    readout=None
)


MODEL_PRESETS: Dict[Model, PipelineConfig] = {
    Model.CLASSICAL_RESERVOIR: CLASSICAL_RESERVOIR_PRESET,
    Model.FNN_DISTILLATION: FNN_DISTILLATION_PRESET,
    Model.FNN: FNN_PRESET,
    Model.PASSTHROUGH: PASSTHROUGH_PRESET,
}

__all__ = [
    "PipelineConfig",
    "MODEL_PRESETS",
    "get_model_preset",
]
