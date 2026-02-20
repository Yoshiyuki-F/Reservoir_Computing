"""Models package."""

from .distillation import DistillationModel, DistillationFactory
from .factory import ModelFactory
from .nn.base import BaseFlaxModel, BaseModel
from .nn.fnn import FNNModel, FNN
from .reservoir import ReservoirFactory, Reservoir, ClassicalReservoir
from .presets import (
    PipelineConfig,
    get_model_preset,
    DistillationConfig,
)

__all__ = [
    "FNN",
    "BaseFlaxModel",
    "FNNModel",
    "BaseModel",
    "ModelFactory",
    "DistillationModel",
    "DistillationFactory",
    "ReservoirFactory",
    "Reservoir",
    "ClassicalReservoir",
    "PipelineConfig",
    "get_model_preset",
    "DistillationConfig",
]
