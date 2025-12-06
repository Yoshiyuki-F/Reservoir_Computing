"""Models package."""

from .distillation import DistillationModel, DistillationFactory
from .factory import ModelFactory
from .nn.base import BaseFlaxModel, BaseModel
from .nn.fnn import FNNModel, FNN
from .nn.rnn import RNNModel, SimpleRNN
from .sequential import SequentialModel
from .reservoir import ReservoirFactory, Reservoir, ClassicalReservoir
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
    "DistillationFactory",
    "SequentialModel",
    "ReservoirFactory",
    "Reservoir",
    "ClassicalReservoir",
    "ModelConfig",
    "MODEL_PRESETS",
    "MODEL_REGISTRY",
    "get_model_preset",
    "DistillationConfig",
]
