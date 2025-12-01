"""Training presets and utilities."""

from .presets import (
    TRAINING_PRESETS,
    TRAINING_REGISTRY,
    TrainingConfig,
    get_training_preset,
    normalize_training_name,
)

__all__ = [
    "TRAINING_PRESETS",
    "TRAINING_REGISTRY",
    "TrainingConfig",
    "get_training_preset",
    "normalize_training_name",
]
