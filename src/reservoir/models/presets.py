

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
    use_intercept=False,
    lambda_candidates=tuple(np.logspace(-12, 3, 30).tolist())
)

DEFAULT_POLY_SQUARE_ONLY_READOUT = PolyRidgeReadoutConfig(
    use_intercept=False,
    lambda_candidates=tuple(np.logspace(-12, 3, 30).tolist()),
    degree=2,
    mode="square_only",
)

DEFAULT_POLY_INTERACTION_ONLY_READOUT = PolyRidgeReadoutConfig(
    use_intercept=False,
    lambda_candidates=tuple(np.logspace(-12, 3, 30).tolist()),
    degree=2,
    mode="interaction_only",
)

DEFAULT_FNN_READOUT = FNNReadoutConfig(hidden_layers=(100,100))


"=============================================Classification Presets============================================"

# MINMAX_MNIST = MinMaxScalerConfig(
#     # feature_min=-0.7675280665952444, #100
#     feature_min=-0.6754946253854848,  # 1200
#     # feature_max=0.35849784076318864, #100
#     feature_max=0.8288112006441126,  # 1200
# )


feature_min, feature_max =-0.6754946253854848, 0.8288112006441126 # 1200

#1200
bound = 1.0
maxminusmin:float = feature_max - feature_min
maxplusmin:float = feature_max + feature_min
scale = maxminusmin / (2.0 * bound)
rs_denom = 2.0 * bound - maxminusmin
rs = maxplusmin / rs_denom if rs_denom != 0 else 0.0

BAS_MNIST = BoundedAffineScalerConfig(
    # scale=0.5630129536792166,  # 100
    scale=scale,  # 1200
    # relative_shift=-0.4680118430007132, # 100
    relative_shift=rs, # 1200
    bound=1.0,
)

RP_MNIST = RandomProjectionConfig(
    n_units=1200,
    # input_scale=0.3543930218531782, #100
    input_scale=0.3478958243673553,  # 1200
    # input_connectivity=0.21745075681282766, #100
    input_connectivity=0.32024990697532185, # 1200
    # bias_scale=0.1725142451754484, #100
    bias_scale= 0.9911807193106197, # 1200
    seed=1,
)


###------Dynamics-------------------------------------------------

CLASSICAL_RESERVOIR_DYNAMICS = ClassicalReservoirConfig(
    # spectral_radius= 1.921291918880454, #100
    spectral_radius= 1.4707341636189577,  # 1200
    # leak_rate= 0.36449529864842045, #100
    leak_rate= 0.5078438853580478, #1200
    # rc_connectivity=0.6784641706491135, #100
    rc_connectivity=0.0760855941265183,  # 1200

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
    preprocess=BAS_MNIST,
    projection=RP_MNIST,
    model=CLASSICAL_RESERVOIR_DYNAMICS,
    # readout=FNNReadoutConfig(hidden_layers=(1000, 1000))
    readout=DEFAULT_RIDGE_READOUT
    # readout=DEFAULT_POLY_SQUARE_ONLY_READOUT
    # readout=DEFAULT_POLY_INTERACTION_ONLY_READOUT
)

'''
uv run python -m reservoir.cli.main --model fnn_distillation_classical --dataset mnist
'''
FNN_DISTILLATION_CLASSICAL_PRESET = PipelineConfig(
    name="fnn_distillation_classical",
    model_type=Model.FNN_DISTILLATION_CLASSICAL,
    description="Feedforward Neural Network with Classical Reservoir Distillation",
    preprocess=BAS_MNIST,
    projection=RP_MNIST,
    model=DistillationConfig(
        teacher=CLASSICAL_RESERVOIR_DYNAMICS,
        student=FNNConfig(
            hidden_layers=(),
        ),
    ),
    readout=DEFAULT_RIDGE_READOUT
)


'''
uv run python -m reservoir.cli.main --model fnn --dataset mnist
'''
FNN_PRESET = PipelineConfig(
    name="fnn",
    model_type=Model.FNN,
    description="Feedforward Neural Network (FNN)",
    preprocess=ZeroToOne,
    projection=None,
    model=FNNConfig(
        hidden_layers=(30,),
    ),
    readout=None
)


# -----------------------------------------------------------------------------
# Quantum Reservoir Definitions
# -----------------------------------------------------------------------------
#
# min, max, bound= -0.23541677149636459, 0.1677064135326006, np.pi
# scale = (max - min) / 2* bound
# relative_shift = (max + min)/2 * bound - (max - min)

s, r, f, lr = 0.3084006355114488, -0.01206032906534976, 3.196929844938574, 0.15402317414946048 #6
# s, r, f, lr = 0.6090938771390537, 0.0, 3.141592653589793,  0.1616784879347744 # 10
QUANTUM_BAPCA = BoundedAffinePCAConfig(
    n_units=6,
    scale=s,
    relative_shift=r,
    bound=math.pi,
)


# Quantum reservoir dynamics (Classification - MEAN aggregation)
QUANTUM_RESERVOIR_DYNAMICS = QuantumReservoirConfig(
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
)

### -----------------------------------------------------------------------------
"""
uv run python -m reservoir.cli.main --model quantum_reservoir --dataset mnist
"""

QUANTUM_RESERVOIR_PRESET = PipelineConfig(
    name="quantum_reservoir",
    model_type=Model.QUANTUM_RESERVOIR,
    description="Quantum Gate-Based Reservoir Computing",
    preprocess=StandardScalerConfig(),
    projection=QUANTUM_BAPCA,
    model=QUANTUM_RESERVOIR_DYNAMICS,
    readout=DEFAULT_RIDGE_READOUT
    # readout=DEFAULT_POLY_SQUARE_ONLY_READOUT
    # readout=DEFAULT_POLY_INTERACTION_ONLY_READOUT
)


'''
uv run python -m reservoir.cli.main --model fnn_distillation_quantum --dataset mnist
'''
FNN_DISTILLATION_QUANTUM_PRESET = PipelineConfig(
    name="fnn_distillation_quantum",
    model_type=Model.FNN_DISTILLATION_QUANTUM,
    description="Feedforward Neural Network with Quantum Reservoir Distillation",
    preprocess=StandardScalerConfig(),
    projection=QUANTUM_BAPCA,
    model=DistillationConfig(
        teacher=QUANTUM_RESERVOIR_DYNAMICS,
        student=FNNConfig(
            hidden_layers=(),
        ),
    ),
    readout=DEFAULT_RIDGE_READOUT,
)

'''
uv run python -m reservoir.cli.main --model passthrough --dataset mnist
'''

PASSTHROUGH_PRESET = PipelineConfig(
    name="passthrough",
    model_type=Model.PASSTHROUGH,
    description="Passthrough model (Projection -> Aggregation, no dynamics)",
    # preprocess=BAS_MNIST,
    # projection=RP_MNIST,
    preprocess=StandardScalerConfig(),
    projection=QUANTUM_BAPCA,
    model=PassthroughConfig(
        aggregation=AggregationMode.MEAN,
    ),
    readout= DEFAULT_POLY_SQUARE_ONLY_READOUT
    # readout=DEFAULT_RIDGE_READOUT
)
















"=============================================Time series Presets================================================================================================================"

# -----------------------------------------------------------------------------
# Dynamics Definitions
# -----------------------------------------------------------------------------

TIME_PROJECTION = RandomProjectionConfig(
    n_units=1024,
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
'''
uv run python -m reservoir.cli.main --model classical_reservoir --dataset mackey_glass
'''
n, seed, mn, mx, scale, ic, bs, sr, lr, rc = 64, 42, -1.252115, 0.197144, 0.780619, 0.486849, 0.676372, 1.198058, 0.553195, 0.983015

TIME_CLASSICAL_RESERVOIR_PRESET = PipelineConfig(
    name="classical_reservoir",
    model_type=Model.CLASSICAL_RESERVOIR,
    description="Echo State Network (Classical Reservoir Computing)",
    preprocess=MinMaxScalerConfig(feature_min=mn, feature_max=mx),
    projection=RandomProjectionConfig(
        n_units=1024,
        input_scale=scale,
        input_connectivity=ic,
        bias_scale=bs,
        seed=seed,
    ),
    model=ClassicalReservoirConfig(
        spectral_radius=sr,
        leak_rate= lr,
        rc_connectivity=rc,
        seed=seed,
        aggregation=AggregationMode.SEQUENCE,
    ),
    readout=DEFAULT_RIDGE_READOUT
)

TIME_FNN_DISTILLATION_CLASSICAL_PRESET = PipelineConfig(
    name="fnn_distillation_classical",
    model_type=Model.FNN_DISTILLATION_CLASSICAL,
    description="Feedforward Neural Network with Classical Reservoir Distillation",
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


# --------------------------------------------------------------------------
# fb > 2
# delta, fs, lr = 0.029014891695261672, 2.3162433911393165, 0.829880311841115 #5_42 7.418181818181818
# delta, fs, lr = 0.0552541001499716, 2.6803338998884767, 0.4707800087677259 #5_42 7.5 unstable

# fb<2
# q5 100 trials
delta, fs, lr = 0.04531012160886314, 0.06881757656255799, 0.8558424190253158


'''
uv run python -m reservoir.cli.main --model quantum_reservoir --dataset lorenz
'''
TIME_QUANTUM_RESERVOIR_PRESET = PipelineConfig(
    name="quantum_reservoir",
    model_type=Model.QUANTUM_RESERVOIR,
    description="Quantum Gate-Based Reservoir Computing (Time Series)",
    preprocess=MinMaxScalerConfig(feature_min=-9.735743764947846e-05, feature_max=delta),
    projection=None, # No projection — MinMaxScaler output goes directly to R-gate
    model=QuantumReservoirConfig(
        n_qubits=11,
        n_layers=1,
        seed=41,
        aggregation=AggregationMode.SEQUENCE,
        feedback_scale=fs,    # a_fb: R gate feedback scaling (paper default)
        leak_rate=lr,         # Leaky integrator rate (tunable by optimizer)
        measurement_basis="Z+ZZ",
        noise_type="clean",
        noise_prob=0.0,
        readout_error=0.0,
        n_trajectories=0,
        use_reuploading=True,
    ),
    readout=DEFAULT_POLY_SQUARE_ONLY_READOUT,
)


"============================================================================================"

# Dispatch table based on (Model, is_classification)
# If entry matches, use specific preset. Else fallback to MODEL_PRESETS.
SPECIFIC_PRESETS: dict[tuple[Model, bool], PipelineConfig] = {
    (Model.CLASSICAL_RESERVOIR, True): CLASSICAL_RESERVOIR_PRESET,
    (Model.FNN, True): FNN_PRESET,
    (Model.FNN_DISTILLATION_CLASSICAL, True): FNN_DISTILLATION_CLASSICAL_PRESET,
    (Model.FNN_DISTILLATION_QUANTUM, True): FNN_DISTILLATION_QUANTUM_PRESET,
    (Model.PASSTHROUGH, True): PASSTHROUGH_PRESET,
    (Model.QUANTUM_RESERVOIR, True): QUANTUM_RESERVOIR_PRESET,

    (Model.CLASSICAL_RESERVOIR, False): TIME_CLASSICAL_RESERVOIR_PRESET,
    (Model.FNN, False): WINDOWED_FNN_PRESET,
    (Model.PASSTHROUGH, False): TIME_PASSTHROUGH_PRESET,
    (Model.FNN_DISTILLATION_CLASSICAL, False): TIME_FNN_DISTILLATION_CLASSICAL_PRESET,
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
