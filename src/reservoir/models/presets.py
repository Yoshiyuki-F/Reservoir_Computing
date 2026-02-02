"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/presets.py
Central registry for model presets.
SSOT: all default hyperparameters live in these dataclasses.
"""
from __future__ import annotations

import numpy as np

from reservoir.core.identifiers import AggregationMode, Model, Dataset
from reservoir.models.config import (
    StandardScalerConfig,
    MaxScalerConfig,
    RandomProjectionConfig,
    CenterCropProjectionConfig,
    ClassicalReservoirConfig,
    DistillationConfig,
    FNNConfig,
    PipelineConfig,
    RidgeReadoutConfig, FNNReadoutConfig, PassthroughConfig,
    QuantumReservoirConfig, ResizeProjectionConfig, PolynomialProjectionConfig,
)
from reservoir.data.presets import get_dataset_preset 




# -----------------------------------------------------------------------------
# Definitions
# -----------------------------------------------------------------------------
#---------------------------STEP 3--------------------------------------------------
RP = RandomProjectionConfig(
    n_units=1000,
    input_scale=0.6,
    input_connectivity=0.1,
    bias_scale=1.0,
    seed=1,
)

CCP = CenterCropProjectionConfig(
    n_units=16,  # This becomes n_qubits for quantum reservoir
)

RES = ResizeProjectionConfig(
    n_units=16,
)

POLY = PolynomialProjectionConfig(
    degree=4,
    include_bias=False,
)
#-----------------------------STEP 7-------------------------------------------------------


DEFAULT_RIDGE_READOUT = RidgeReadoutConfig(
    use_intercept=True,
    lambda_candidates=tuple(np.logspace(-12, 3, 30).tolist())
)

DEFAULT_FNN_READOUT = FNNReadoutConfig(hidden_layers=(1000,))


"=============================================Classification Presets============================================"

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
    preprocess=MaxScalerConfig(),
    projection=POLY,
    model=CLASSICAL_RESERVOIR_DYNAMICS,
    readout=DEFAULT_RIDGE_READOUT
)

FNN_DISTILLATION_PRESET = PipelineConfig(
    name="fnn_distillation",
    model_type=Model.FNN_DISTILLATION,
    description="Feedforward Neural Network with Reservoir Distillation",
    preprocess=MaxScalerConfig(),
    projection=RP,
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
    preprocess=MaxScalerConfig(),
    projection=RP,
    model=PassthroughConfig(
        aggregation=AggregationMode.MEAN,
    ),
    readout=DEFAULT_RIDGE_READOUT
)


FNN_PRESET = PipelineConfig(
    name="fnn",
    model_type=Model.FNN,
    description="Feedforward Neural Network (FNN)",
    preprocess=MaxScalerConfig(),
    projection=None,
    model=FNNConfig(
        hidden_layers=(100,),
    ),
    readout=None
)


# -----------------------------------------------------------------------------
# Quantum Reservoir Definitions
# -----------------------------------------------------------------------------

# Quantum reservoir dynamics (Classification - MEAN aggregation)
QUANTUM_RESERVOIR_DYNAMICS = QuantumReservoirConfig(
    n_layers=3,
    seed=41,
    aggregation=AggregationMode.MEAN,
    feedback_scale=1.2,
    measurement_basis="Z+ZZ",
    encoding_strategy="Rx",
    noise_type="clean",
    noise_prob=0.0,
    readout_error=0.0,
    n_trajectories=0,
    use_remat=False,
    use_reuploading=True,
    precision="complex64",
)


# Time-series Quantum Reservoir (SEQUENCE aggregation)
TIME_QUANTUM_RESERVOIR_DYNAMICS = QuantumReservoirConfig(
    n_layers=3, #4 with feedback 1.7
    seed=41,
    aggregation=AggregationMode.SEQUENCE,
    feedback_scale=1.2,
    measurement_basis="Z+ZZ",
    encoding_strategy="Rx",
    noise_type="clean",
    noise_prob=0.0,
    readout_error=0.0,
    n_trajectories=0,
    use_remat=False,
    use_reuploading=True,
    precision="complex64",
)

QUANTUM_RESERVOIR_PRESET = PipelineConfig(
    name="quantum_reservoir",
    model_type=Model.QUANTUM_RESERVOIR,
    description="Quantum Gate-Based Reservoir Computing",
    preprocess=MaxScalerConfig(),
    projection=RES,
    model=QUANTUM_RESERVOIR_DYNAMICS,
    readout=DEFAULT_RIDGE_READOUT,
)

TIME_QUANTUM_RESERVOIR_PRESET = PipelineConfig(
    name="quantum_reservoir",
    model_type=Model.QUANTUM_RESERVOIR,
    description="Quantum Gate-Based Reservoir Computing (Time Series)",
    preprocess=StandardScalerConfig(),
    projection=RES,  # Reuse the 4-qubit projection
    model=TIME_QUANTUM_RESERVOIR_DYNAMICS,
    readout=DEFAULT_RIDGE_READOUT,
)

"=============================================Time series Presets============================================"

# -----------------------------------------------------------------------------
# Definitions
# -----------------------------------------------------------------------------


TIME_PROJECTION = RandomProjectionConfig(
    n_units=100,
    input_scale=0.2,
    input_connectivity=1.0,
    bias_scale=0.1,
    seed=1,
)

TIME_RESERVOIR_DYNAMICS = ClassicalReservoirConfig(
    spectral_radius=1,
    leak_rate=0.4, #best for n ,not for n=1000
    rc_connectivity=0.02, #best
    seed=42,
    aggregation=AggregationMode.SEQUENCE,
)

# -------------------------------------------------------------------------------


TIME_CLASSICAL_RESERVOIR_PRESET = PipelineConfig(
    name="classical_reservoir",
    model_type=Model.CLASSICAL_RESERVOIR,
    description="Echo State Network (Classical Reservoir Computing)",
    preprocess=StandardScalerConfig(),
    projection=TIME_PROJECTION,
    model=TIME_RESERVOIR_DYNAMICS,
    readout=DEFAULT_RIDGE_READOUT
)

TIME_FNN_DISTILLATION_PRESET = PipelineConfig(
    name="fnn_distillation",
    model_type=Model.FNN_DISTILLATION,
    description="Feedforward Neural Network with Reservoir Distillation",
    preprocess=StandardScalerConfig(),
    projection=TIME_PROJECTION,
    model=DistillationConfig(
        teacher=TIME_RESERVOIR_DYNAMICS,
        student=FNNConfig(
            hidden_layers=(500, 500),
            window_size=64,  # This enables TimeDelayEmbedding adapter
        ),
    ),
    readout=DEFAULT_RIDGE_READOUT
)


TIME_PASSTHROUGH_PRESET = PipelineConfig(
    name="passthrough",
    model_type=Model.PASSTHROUGH,
    description="Passthrough model (Projection -> Aggregation, no dynamics)",
    preprocess=StandardScalerConfig(),
    projection=TIME_PROJECTION,
    model=PassthroughConfig(
        aggregation=AggregationMode.SEQUENCE,
    ),
    readout=DEFAULT_RIDGE_READOUT
)



WINDOWED_FNN_PRESET = PipelineConfig(
    name="windowed_fnn",
    model_type=Model.FNN,  # Same model type, different config
    description="FNN with sliding window embedding for time series regression",
    preprocess=StandardScalerConfig(),
    projection=None,  # No projection needed
    model=FNNConfig(
        hidden_layers=(100, 100),
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
    (Model.QUANTUM_RESERVOIR, True): QUANTUM_RESERVOIR_PRESET,

    (Model.CLASSICAL_RESERVOIR, False): TIME_CLASSICAL_RESERVOIR_PRESET,
    (Model.FNN, False): WINDOWED_FNN_PRESET,
    (Model.PASSTHROUGH, False): TIME_PASSTHROUGH_PRESET,
    (Model.FNN_DISTILLATION, False): TIME_FNN_DISTILLATION_PRESET,
    (Model.QUANTUM_RESERVOIR, False): TIME_QUANTUM_RESERVOIR_PRESET,
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
