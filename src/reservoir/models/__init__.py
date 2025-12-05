"""Models package."""

from .distillation import DistillationModel
from .factory import ModelFactory
from .nn.base import BaseFlaxModel, BaseModel
from .nn.fnn import FNNModel, FNN
from .nn.rnn import RNNModel, SimpleRNN
from .reservoir import ReservoirModel, ReservoirFactory, Reservoir, ClassicalReservoir
from .presets import (
    ModelConfig,
    MODEL_PRESETS,
    MODEL_REGISTRY,
    get_model_preset,
    DistillationConfig,
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
    "ModelConfig",
    "MODEL_PRESETS",
    "MODEL_REGISTRY",
    "get_model_preset",
    "DistillationConfig",
]
