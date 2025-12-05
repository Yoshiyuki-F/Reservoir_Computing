"""Models package."""

from .distillation import DistillationModel
from .factories import FlaxModelFactory
from .nn.base import BaseFlaxModel, BaseModel
from .nn.fnn import FNNModel, FNN
from .nn.rnn import RNNModel, SimpleRNN
from .presets import (
    ModelPreset,
    ReservoirConfig,
    DistillationConfig,
    MODEL_PRESETS,
    MODEL_REGISTRY,
    get_model_preset,
    normalize_model_name,
)

__all__ = [
    "SimpleRNN",
    "FNN",
    "BaseFlaxModel",
    "FNNModel",
    "RNNModel",
    "BaseModel",
    "FlaxModelFactory",
    "DistillationModel",
    "ModelPreset",
    "ReservoirConfig",
    "DistillationConfig",
    "MODEL_PRESETS",
    "MODEL_REGISTRY",
    "get_model_preset",
    "normalize_model_name",
]
