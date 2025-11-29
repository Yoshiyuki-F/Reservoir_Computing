"""Core components shared across models.

This package hosts infrastructure components (preprocessing, readouts, RNG helpers)
that are intentionally decoupled from any specific model implementation.
"""

from .preprocess.scaler import FeatureScaler  # noqa: F401
from .preprocess.design_matrix import DesignMatrixBuilder  # noqa: F401
from .preprocess.aggregator import aggregate_states, StateAggregator  # noqa: F401
from .preprocess.pipeline import TransformerSequence  # noqa: F401
from .readout.ridge import RidgeRegression  # noqa: F401

__all__ = [
    "FeatureScaler",
    "DesignMatrixBuilder",
    "aggregate_states",
    "StateAggregator",
    "TransformerSequence",
    "RidgeRegression",
]
