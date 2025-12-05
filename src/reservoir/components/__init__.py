"""Core components shared across models.

This package hosts infrastructure components (preprocessing, readouts, RNG helpers)
that are intentionally decoupled from any specific model implementation.
"""

from .projection import InputProjector  # noqa: F401
from .preprocess import (
    FeatureScaler,
    DesignMatrixBuilder,
    DesignMatrix,
    aggregate_states,
    TransformerSequence,
    StateAggregator,
)  # noqa: F401
from .readout.ridge import RidgeRegression  # noqa: F401

__all__ = [
    "InputProjector",
    "FeatureScaler",
    "DesignMatrixBuilder",
    "DesignMatrix",
    "aggregate_states",
    "TransformerSequence",
    "StateAggregator",
    "RidgeRegression",
]
