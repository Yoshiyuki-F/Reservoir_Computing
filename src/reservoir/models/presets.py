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



"=============================================Classification Presets============================================"

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

CLASSICAL_RESERVOIR_DYNAMICS = ClassicalReservoirConfig(
    spectral_radius=1.3,
    leak_rate=0.2,
    rc_connectivity=0.9,
    seed=42,
    aggregation=AggregationMode.MEAN,
)

# ------------------------------------------------------------------------------------


CLASSICAL_RESERVOIR_PRESET = PipelineConfig(
    name="classical_reservoir",
    model_type=Model.CLASSICAL_RESERVOIR,
    description="Echo State Network (Classical Reservoir Computing)",
    preprocess=DEFAULT_PREPROCESS,
    projection=DEFAULT_PROJECTION,
    model=CLASSICAL_RESERVOIR_DYNAMICS,
    readout=DEFAULT_RIDGE_READOUT
)

FNN_DISTILLATION_PRESET = PipelineConfig(
    name="fnn_distillation",
    model_type=Model.FNN_DISTILLATION,
    description="Feedforward Neural Network with Reservoir Distillation",
    preprocess=DEFAULT_PREPROCESS,
    projection=DEFAULT_PROJECTION,
    model=DistillationConfig(
        teacher=CLASSICAL_RESERVOIR_DYNAMICS,
        student=FNNConfig(
            hidden_layers=(100, ),
        ),
    ),
    readout=DEFAULT_RIDGE_READOUT
)

PASSTHROUGH_PRESET = PipelineConfig(
    name="passthrough",
    model_type=Model.PASSTHROUGH,
    description="Passthrough model (Projection -> Aggregation, no dynamics)",
    preprocess=DEFAULT_PREPROCESS,
    projection=DEFAULT_PROJECTION,
    model=PassthroughConfig(
        aggregation=AggregationMode.MEAN,
    ),
    readout=DEFAULT_RIDGE_READOUT
)


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

"=============================================Time series Presets============================================"

# -----------------------------------------------------------------------------
# Definitions
# -----------------------------------------------------------------------------

DEFAULT_PROJECTION_REGRESSION = ProjectionConfig(
    n_units=1000,
    input_scale=0.2,
    input_connectivity=1.0,
    bias_scale=0.1,
    seed=1,
)

# -------------------------------------------------------------------------------


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

TIME_PASSTHROUGH_PRESET = PipelineConfig(
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



WINDOWED_FNN_PRESET = PipelineConfig(
    name="windowed_fnn",
    model_type=Model.FNN,  # Same model type, different config
    description="FNN with sliding window embedding for time series regression",
    preprocess=PreprocessingConfig(
        method=Preprocessing.STANDARD_SCALER,
        poly_degree=1,
    ),
    projection=None,  # No projection needed
    model=FNNConfig(
        hidden_layers=(1000, 1000),
        window_size=64,  # This enables TimeDelayEmbedding adapter
    ),
    readout=None,  # FNN is end-to-end
)

"============================================================================================"

# Dispatch table based on (Model, is_classification)
# If entry matches, use specific preset. Else fallback to MODEL_PRESETS.
SPECIFIC_PRESETS = {
    (Model.CLASSICAL_RESERVOIR, True): CLASSICAL_RESERVOIR_PRESET,
    (Model.FNN, True): FNN_PRESET,
    (Model.FNN_DISTILLATION, True): FNN_DISTILLATION_PRESET,
    (Model.PASSTHROUGH, True): PASSTHROUGH_PRESET,

    (Model.CLASSICAL_RESERVOIR, False): TIME_CLASSICAL_RESERVOIR_PRESET,
    (Model.FNN, False): WINDOWED_FNN_PRESET,
    (Model.PASSTHROUGH, False): TIME_PASSTHROUGH_PRESET,
}

def get_model_preset(model: Model, dataset: Dataset) -> PipelineConfig:
    """Retrieves a model preset by enum key with task-aware dispatch."""
    ds_preset = get_dataset_preset(dataset)
    is_classification = ds_preset.classification
    
    if (model, is_classification) in SPECIFIC_PRESETS:
        return SPECIFIC_PRESETS[(model, is_classification)]
    else:
        return None

__all__ = [
    "PipelineConfig",
    "get_model_preset",
]
