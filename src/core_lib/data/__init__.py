"""Core data generation and preparation utilities.

This package hosts reusable dataset generators and preparation helpers
that are shared across multiple experiment pipelines.
"""

from .data_preparation import (
    ExperimentDataset,
    ExperimentDatasetClassification,
    prepare_experiment_data,
)
from .generators import (
    generate_sine_data,
    generate_lorenz_data,
    generate_mackey_glass_data,
    generate_mnist_sequence_data,
)
from .registry import DatasetRegistry  # noqa: F401
# Import loaders for side-effect registration
from . import loaders  # noqa: F401

__all__ = [
    "ExperimentDataset",
    "ExperimentDatasetClassification",
    "prepare_experiment_data",
    "generate_sine_data",
    "generate_lorenz_data",
    "generate_mackey_glass_data",
    "generate_mnist_sequence_data",
    "DatasetRegistry",
]
