"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/presets.py
Central registry for model presets.
SSOT: all default hyperparameters live in these dataclasses.
"""
from __future__ import annotations

import numpy as np

from reservoir.layers.aggregation import AggregationMode
from reservoir.models.identifiers import Model
from reservoir.models.config import (
    StandardScalerConfig,
    MinMaxScalerConfig,
    RandomProjectionConfig,
    CenterCropProjectionConfig,
    PCAProjectionConfig,
    ClassicalReservoirConfig,
    DistillationConfig,
    FNNConfig,
    PipelineConfig,
    RidgeReadoutConfig, FNNReadoutConfig, PassthroughConfig, PolyRidgeReadoutConfig,
    QuantumReservoirConfig, ResizeProjectionConfig, PolynomialProjectionConfig, AffineScalerConfig
)
from reservoir.data.presets import get_dataset_preset 
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reservoir.data.identifiers import Dataset




# -----------------------------------------------------------------------------
# Definitions
# -----------------------------------------------------------------------------
#---------------------------STEP 2--------------------------------------------------

ZeroToOne = MinMaxScalerConfig(
    feature_min=0.0,
    feature_max=1.0,
)

MinusOneToOne = MinMaxScalerConfig(
    feature_min=-1.0,
    feature_max=1.0,
)

#---------------------------STEP 3--------------------------------------------------
RP = RandomProjectionConfig(
    n_units=100,
    input_scale=1.0,
    input_connectivity=0.11458754901458218,
    bias_scale= 0.8295811429210161,
    seed=1,
)



CCP = CenterCropProjectionConfig(
    n_units=10,  # This becomes n_qubits for quantum reservoir
)

RES = ResizeProjectionConfig(
    n_units=10,
)

POLY = PolynomialProjectionConfig(
    degree=4,
    include_bias=False,
)

PCA = PCAProjectionConfig(
    n_units=16,
    input_scaler = 0.08, # 0.05 67 0.07 68.6 0.08 68.77 0.09 68.29 0.1 68.32 0.15 65.4 0.2 64.4 0.3 62.9% 0.4 61% 0.8 56%
)

#-----------------------------STEP 7-------------------------------------------------------


DEFAULT_RIDGE_READOUT = RidgeReadoutConfig(
    use_intercept=True,
    lambda_candidates=tuple(np.logspace(-12, 3, 30).tolist())
)

DEFAULT_POLY_RIDGE_READOUT = PolyRidgeReadoutConfig(
    use_intercept=True,
    lambda_candidates=tuple(np.logspace(-12, 3, 30).tolist()),
    degree=2,
    mode="square_only",
)


DEFAULT_FNN_READOUT = FNNReadoutConfig(hidden_layers=(100,))


"=============================================Classification Presets============================================"

# -----------------------------------------------------------------------------
# Dynamics Definitions
# -----------------------------------------------------------------------------
CLASSICAL_RESERVOIR_DYNAMICS = ClassicalReservoirConfig(
    spectral_radius=1.45,
    leak_rate= 0.66,
    rc_connectivity=0.457758485877939,
    seed=42,
    aggregation=AggregationMode.MEAN,
)


# -----------------------------------------------------------------------------

CLASSICAL_RESERVOIR_PRESET = PipelineConfig(
    name="classical_reservoir",
    model_type=Model.CLASSICAL_RESERVOIR,
    description="Echo State Network (Classical Reservoir Computing)",
    preprocess=MinMaxScalerConfig(
        feature_min=-0.07768410112268466,
        feature_max=0.08160917176536134,
    ),
    projection=RP,
    model=CLASSICAL_RESERVOIR_DYNAMICS,
    readout=DEFAULT_RIDGE_READOUT
)

FNN_DISTILLATION_PRESET = PipelineConfig(
    name="fnn_distillation",
    model_type=Model.FNN_DISTILLATION,
    description="Feedforward Neural Network with Reservoir Distillation",
    preprocess=ZeroToOne,
    projection=RP,
    model=DistillationConfig(
        teacher=CLASSICAL_RESERVOIR_DYNAMICS,
        student=FNNConfig(
            hidden_layers=(10000, ),
        ),
    ),
    readout=DEFAULT_RIDGE_READOUT
)

PASSTHROUGH_PRESET = PipelineConfig(
    name="passthrough",
    model_type=Model.PASSTHROUGH,
    description="Passthrough model (Projection -> Aggregation, no dynamics)",
    preprocess=ZeroToOne,
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
    preprocess=ZeroToOne,
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
    feedback_scale=0.0,    # a_fb=0.0 means no feedback (pure feedforward mode)
    measurement_basis="Z+ZZ",
    noise_type="clean",
    noise_prob=0.0,
    readout_error=0.0,
    n_trajectories=0,
    use_remat=False,
    use_reuploading=True,
    precision="complex64",
)

# -----------------------------------------------------------------------------

QUANTUM_RESERVOIR_PRESET = PipelineConfig(
    name="quantum_reservoir",
    model_type=Model.QUANTUM_RESERVOIR,
    description="Quantum Gate-Based Reservoir Computing",
    preprocess=ZeroToOne,
    projection=PCA,
    model=QUANTUM_RESERVOIR_DYNAMICS,
    readout=DEFAULT_RIDGE_READOUT,
)

max_input_scaler = 3.314782572029597
TIME_QUANTUM_RESERVOIR_PRESET = PipelineConfig(
    name="quantum_reservoir",
    model_type=Model.QUANTUM_RESERVOIR,
    description="Quantum Gate-Based Reservoir Computing (Time Series)",
    preprocess=AffineScalerConfig(input_scale=max_input_scaler/0.9268, shift=-0.4015*max_input_scaler/0.9268),
    projection=None, # No projection — MinMaxScaler output goes directly to R-gate
    model=QuantumReservoirConfig(
        n_qubits=16,
        n_layers=3,
        seed=41,
        aggregation=AggregationMode.SEQUENCE,
        feedback_scale=2.0040817003461666,    # a_fb: R gate feedback scaling (paper default)
        measurement_basis="Z+ZZ",
        noise_type="clean",
        noise_prob=0.0,
        readout_error=0.0,
        n_trajectories=0,
        use_remat=False,
        use_reuploading=True,
        precision="complex128",
    ),
    readout=DEFAULT_POLY_RIDGE_READOUT,
)

"=============================================Time series Presets================================================================================================================"

# -----------------------------------------------------------------------------
# Dynamics Definitions
# -----------------------------------------------------------------------------

TIME_PROJECTION = RandomProjectionConfig(
    n_units=400,
    input_scale=0.1,
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
    projection=RandomProjectionConfig(
        n_units=100,
        input_scale=1.1900256705783303,
        input_connectivity=0.16412483680491705,
        bias_scale=1,
        seed=1,
    ),
    model=ClassicalReservoirConfig(
        spectral_radius=1.2275524643139968,
        leak_rate= 0.4176226904512959,
        rc_connectivity=0.6014914261660489,
        seed=42,
        aggregation=AggregationMode.SEQUENCE,
    ),
    readout=DEFAULT_RIDGE_READOUT
)

TIME_FNN_DISTILLATION_PRESET = PipelineConfig(
    name="fnn_distillation",
    model_type=Model.FNN_DISTILLATION,
    description="Feedforward Neural Network with Reservoir Distillation",
    preprocess=StandardScalerConfig(),
    projection=RP,
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
SPECIFIC_PRESETS: dict[tuple[Model, bool], PipelineConfig] = {
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
    if ds_preset is None:
        raise ValueError(f"No dataset preset found for dataset: {dataset}")
    is_classification = getattr(ds_preset, "classification", False)
    
    if (model, is_classification) in SPECIFIC_PRESETS:
        return SPECIFIC_PRESETS[(model, is_classification)]
    else:
        raise ValueError(f"No preset found for model {model} with classification={is_classification}")

__all__ = [
    "PipelineConfig",
    "get_model_preset",
]
