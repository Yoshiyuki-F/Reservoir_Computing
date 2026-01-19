"""Models package."""

from .distillation import DistillationModel, DistillationFactory
from .factory import ModelFactory
from .nn.base import BaseFlaxModel, BaseModel
from .nn.fnn import FNNModel, FNN
from .nn.rnn import RNNModel, SimpleRNN
from .reservoir import ReservoirFactory, Reservoir, ClassicalReservoir
from .presets import (
    PipelineConfig,
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
    "ReservoirFactory",
    "Reservoir",
    "ClassicalReservoir",
    "PipelineConfig",
    "MODEL_PRESETS",
    "get_model_preset",
    "DistillationConfig",
]
