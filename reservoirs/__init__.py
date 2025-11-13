"""
Core reservoir utilities shared across classical and quantum implementations.

This package hosts infrastructure components (preprocessing, readouts, RNG helpers)
that are intentionally decoupled from any specific reservoir computer.
"""

from .preprocess.scaler import FeatureScaler  # noqa: F401
from .preprocess.design_matrix import DesignMatrixBuilder  # noqa: F401
from .preprocess.aggregator import aggregate_states  # noqa: F401
from .readout.base import BaseReadout, ReadoutResult  # noqa: F401
from .readout.ridge_numpy import RidgeReadoutNumpy  # noqa: F401

__all__ = [
    "FeatureScaler",
    "DesignMatrixBuilder",
    "aggregate_states",
    "BaseReadout",
    "ReadoutResult",
    "RidgeReadoutNumpy",
]
