"""Models package."""

from .distillation import DistillationModel
from .factory import ModelFactory
from .nn.base import BaseFlaxModel, BaseModel
from .nn.fnn import FNNModel, FNN
from .nn.rnn import RNNModel, SimpleRNN
from .reservoir import ReservoirModel, ReservoirFactory, Reservoir, ClassicalReservoir
from .presets import (
    ModelPreset,
    ReservoirConfig,
    DistillationConfig,
    MODEL_PRESETS,
    MODEL_REGISTRY,
    get_model_preset,
)

__all__ = [
    "SimpleRNN",
    "FNN",
    "BaseFlaxModel",
    "FNNModel",
    "RNNModel",
    "BaseModel",
    "ModelFactory",
    "DistillationModel",
    "ReservoirModel",
    "ReservoirFactory",
    "Reservoir",
    "ClassicalReservoir",
    "ModelPreset",
    "ReservoirConfig",
    "DistillationConfig",
    "MODEL_PRESETS",
    "MODEL_REGISTRY",
    "get_model_preset",
]
