"""FNN model and configuration subpackage."""

from .model import FNN
from .config import FNNModelConfig, FNNTrainingConfig, FNNPipelineConfig

__all__ = [
    "FNN",
    "FNNModelConfig",
    "FNNTrainingConfig",
    "FNNPipelineConfig",
]

