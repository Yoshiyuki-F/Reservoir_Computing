"""Models package."""

from .factories import FlaxModelFactory
from .nn.base import BaseFlaxModel
from .nn.modules import FNN, SimpleRNN
from .nn.fnn import FNNModel
from .nn.rnn import RNNModel
from .nn.config import (
    FNNModelConfig,
    FNNTrainingConfig,
    FNNPipelineConfig,
    SimpleRNNConfig,
)
from .presets import (
    ModelPreset,
    ReservoirConfig,
    MODEL_PRESETS,
    MODEL_REGISTRY,
    get_model_preset,
    normalize_model_name,
)

__all__ = [
    "SimpleRNN",
    "SimpleRNNConfig",
    "FNN",
    "FNNModelConfig",
    "FNNTrainingConfig",
    "FNNPipelineConfig",
    "BaseFlaxModel",
    "FNNModel",
    "RNNModel",
    "FlaxModelFactory",
    "ModelPreset",
    "ReservoirConfig",
    "MODEL_PRESETS",
    "MODEL_REGISTRY",
    "get_model_preset",
    "normalize_model_name",
]
