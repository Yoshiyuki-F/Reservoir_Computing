"""Core data generation and preparation utilities."""

from .generators import (
    generate_sine_data,
    generate_lorenz_data,
    generate_mackey_glass_data,
    generate_mnist_sequence_data,
)
from .presets import (
    DATASET_PRESETS,
    DATASET_REGISTRY,
    DatasetPreset,
    get_dataset_preset,
    normalize_dataset_name,
)
from .structs import SplitDataset
from .registry import DatasetRegistry  # noqa: F401
from . import loaders  # noqa: F401
from .mnist_loader import get_mnist_datasets, image_to_sequence  # noqa: F401

__all__ = [
    "generate_sine_data",
    "generate_lorenz_data",
    "generate_mackey_glass_data",
    "generate_mnist_sequence_data",
    "DATASET_PRESETS",
    "DATASET_REGISTRY",
    "DatasetPreset",
    "get_dataset_preset",
    "normalize_dataset_name",
    "SplitDataset",
    "DatasetRegistry",
    "get_mnist_datasets",
    "image_to_sequence",
]
