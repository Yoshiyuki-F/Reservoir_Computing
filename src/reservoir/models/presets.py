

"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/presets.py
Central registry for model presets.
SSOT: all default hyperparameters live in these dataclasses.
"""
from __future__ import annotations
import math
import numpy as np


from reservoir.layers.aggregation import AggregationMode
from reservoir.models.identifiers import Model
from reservoir.models.config import (
    StandardScalerConfig,
    MinMaxScalerConfig,
    RandomProjectionConfig,
    CenterCropProjectionConfig,
    PCAProjectionConfig,
    BoundedAffinePCAConfig,
    ClassicalReservoirConfig,
    DistillationConfig,
    FNNConfig,
    PipelineConfig,
    RidgeReadoutConfig, FNNReadoutConfig, PassthroughConfig, PolyRidgeReadoutConfig,
    QuantumReservoirConfig, ResizeProjectionConfig, PolynomialProjectionConfig, AffineScalerConfig,
    BoundedAffineScalerConfig
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


RP_REGRESSION = RandomProjectionConfig(
    n_units=64,
    # input_scale=1.0, #100
    input_scale=0.3478958243673553,  # 1200
    # input_connectivity=0.11458754901458218, #100
    input_connectivity=0.32024990697532185, # 1200
    # bias_scale=0.8295811429210161, #100
    bias_scale= 0.9911807193106197, # 1200
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

MINMAX_MNIST = MinMaxScalerConfig(
    feature_min=-0.7675280665952444, #100
    # feature_min=-0.6754946253854848,  # 1200
    feature_max=0.35849784076318864, #100
    # feature_max=0.8288112006441126,  # 1200
)

RP_MNIST = RandomProjectionConfig(
    n_units=100,
    input_scale=0.3543930218531782, #100
    # input_scale=0.3478958243673553,  # 1200
    input_connectivity=0.21745075681282766, #100
    # input_connectivity=0.32024990697532185, # 1200
    bias_scale=0.1725142451754484, #100
    # bias_scale= 0.9911807193106197, # 1200
    seed=1,
)


CLASSICAL_RESERVOIR_DYNAMICS = ClassicalReservoirConfig(
    spectral_radius= 1.921291918880454, #100
    # spectral_radius= 1.4707341636189577,  # 1200
    leak_rate= 0.36449529864842045, #100
    # leak_rate= 0.5078438853580478, #1200
    rc_connectivity=0.6784641706491135, #100
    # rc_connectivity=0.0760855941265183,  # 1200

    seed=42,
    aggregation=AggregationMode.MEAN,
)


# -----------------------------------------------------------------------------
'''
uv run python -m reservoir.cli.main --model classical_reservoir --dataset mnist
'''
CLASSICAL_RESERVOIR_PRESET = PipelineConfig(
    name="classical_reservoir",
    model_type=Model.CLASSICAL_RESERVOIR,
    description="Echo State Network (Classical Reservoir Computing)",
    preprocess=MINMAX_MNIST,
    projection=RP_MNIST,
    model=CLASSICAL_RESERVOIR_DYNAMICS,
    readout=DEFAULT_RIDGE_READOUT
)

FNN_DISTILLATION_PRESET = PipelineConfig(
    name="fnn_distillation",
    model_type=Model.FNN_DISTILLATION,
    description="Feedforward Neural Network with Reservoir Distillation",
    preprocess=MINMAX_MNIST,
    projection=RP_MNIST,
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
    preprocess=MINMAX_MNIST,
    projection=RP_MNIST,
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
    leak_rate=1.0,         # No leaky integration (backward compatible)
    measurement_basis="Z+ZZ",
    noise_type="clean",
    noise_prob=0.0,
    readout_error=0.0,
    n_trajectories=0,
    use_reuploading=True,
)

### -----------------------------------------------------------------------------
"""
uv run python -m reservoir.cli.main --model quantum_reservoir --dataset mnist
"""
min, max, bound= -0.23541677149636459, 0.1677064135326006, 1
scale = (max - min) / 2* bound
relative_shift = (max + min)/2 * bound - (max - min)

s, r, f, lr = 0.3084006355114488, -0.01206032906534976, 3.196929844938574, 0.15402317414946048
QUANTUM_RESERVOIR_PRESET = PipelineConfig(
    name="quantum_reservoir",
    model_type=Model.QUANTUM_RESERVOIR,
    description="Quantum Gate-Based Reservoir Computing",
    preprocess=StandardScalerConfig(),
    projection=BoundedAffinePCAConfig(
        n_units=10,
        scale=s,
        relative_shift=r,
        bound=math.pi,
    ),
    model=QuantumReservoirConfig(
        n_layers=1,
        seed=41,
        aggregation=AggregationMode.MEAN,
        feedback_scale=f,    # a_fb=0.0 means no feedback (pure feedforward mode)
        leak_rate=lr,         # No leaky integration (backward compatible)
        measurement_basis="Z+ZZ",
        noise_type="clean",
        noise_prob=0.0,
        readout_error=0.0,
        n_trajectories=0,
        use_reuploading=True,
    ),
    readout=DEFAULT_POLY_RIDGE_READOUT,
)

TIME_QUANTUM_RESERVOIR_PRESET = PipelineConfig(
    name="quantum_reservoir",
    model_type=Model.QUANTUM_RESERVOIR,
    description="Quantum Gate-Based Reservoir Computing (Time Series)",
    preprocess=MinMaxScalerConfig(feature_min=0.0, feature_max=0.04387396511208059),
    projection=None, # No projection — MinMaxScaler output goes directly to R-gate
    model=QuantumReservoirConfig(
        n_qubits=6,
        n_layers=7,
        seed=42,
        aggregation=AggregationMode.SEQUENCE,
        feedback_scale=3.288848187732551,    # a_fb: R gate feedback scaling (paper default)
        leak_rate=0.11967302052818608,         # Leaky integrator rate (tunable by optimizer)
        measurement_basis="Z+ZZ",
        noise_type="clean",
        noise_prob=0.0,
        readout_error=0.0,
        n_trajectories=0,
        use_reuploading=True,
    ),
    readout=DEFAULT_POLY_RIDGE_READOUT,
)

"=============================================Time series Presets================================================================================================================"

# -----------------------------------------------------------------------------
# Dynamics Definitions
# -----------------------------------------------------------------------------

TIME_PROJECTION = RandomProjectionConfig(
    n_units=64,
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
    preprocess=MinMaxScalerConfig(feature_min=0.0, feature_max=0.04387396511208059),
    projection=RandomProjectionConfig(
        n_units=1024,
        input_scale=0.5021672479393327,
        input_connectivity=0.48415316598538416,
        bias_scale=1.2287247970196717,
        seed=42,
    ),
    model=ClassicalReservoirConfig(
        spectral_radius=1.190314226578602,
        leak_rate= 0.28133317330437824,
        rc_connectivity=0.5421776698098623,
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
    projection=RP_REGRESSION,
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
