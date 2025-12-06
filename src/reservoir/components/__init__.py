"""Core components shared across models.

This package hosts infrastructure components (preprocessing, readouts, RNG helpers)
that are intentionally decoupled from any specific model implementation.
"""

from .projection import InputProjector  # noqa: F401
from .preprocess import (
    FeatureScaler,
    DesignMatrix,
    TransformerSequence,
    StateAggregator,
)  # noqa: F401
from .readout.ridge import RidgeRegression  # noqa: F401

__all__ = [
    "InputProjector",
    "FeatureScaler",
    "DesignMatrix",
    "TransformerSequence",
    "StateAggregator",
    "RidgeRegression",
]
