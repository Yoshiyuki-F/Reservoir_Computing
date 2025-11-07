"""Model-agnostic data preparation utilities."""

from typing import Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass

import jax.numpy as jnp

from .preprocessing import normalize_data
from .generators import (
    generate_sine_data,
    generate_lorenz_data,
    generate_mackey_glass_data,
    generate_mnist_sequence_data,
)


@dataclass
class ExperimentDataset:
    """Container for prepared experiment data."""
    train_input: jnp.ndarray
    train_target: jnp.ndarray
    test_input: jnp.ndarray
    test_target: jnp.ndarray
    target_mean: float
    target_std: float
    train_size: int


@dataclass
class ExperimentDatasetClassification:
    """Dataset container for classification experiments."""
    train_sequences: jnp.ndarray
    train_labels: jnp.ndarray
    test_sequences: jnp.ndarray
    test_labels: jnp.ndarray
    sequence_mean: Optional[float] = None
    sequence_std: Optional[float] = None
    train_size: int = 0




def prepare_experiment_data(
    config,
    quantum_mode: bool = False,
) -> Union[ExperimentDataset, ExperimentDatasetClassification]:
    """Prepare data for any model type experiment.

    Args:
        config: ExperimentConfig object containing data generation parameters
        quantum_mode: Whether to prepare data for quantum computing (currently unused)

    Returns:
        ExperimentDataset with normalized and split training/test data
    """
    # Get data generation function based on dataset name
    data_generators = {
        'sine_wave': generate_sine_data,
        'lorenz': generate_lorenz_data,
        'mackey_glass': generate_mackey_glass_data,
    }
    if generate_mnist_sequence_data is not None:
        data_generators['mnist'] = generate_mnist_sequence_data

    dataset_name = config.data_generation.name
    task_type = getattr(config.training, "task_type", "timeseries")

    if task_type == "classification":
        if dataset_name != 'mnist':
            raise ValueError(
                f"Classification task_type currently supports only 'mnist' dataset, got '{dataset_name}'"
            )
        if generate_mnist_sequence_data is None:
            raise ImportError("MNIST dataset generation requires torch/torchvision to be installed.")

        sequences, labels = generate_mnist_sequence_data(config.data_generation)
        sequences = jnp.array(sequences, dtype=jnp.float64)
        labels = jnp.array(labels, dtype=jnp.int32)

        # Normalize sequences if requested
        norm_sequences, seq_mean, seq_std = normalize_data(sequences)

        total_length = norm_sequences.shape[0]
        default_train_size = int(0.8 * total_length)
        target_train_size = getattr(config.training, "train_size", None)
        if target_train_size is not None:
            train_size = int(total_length * target_train_size)
        else:
            train_size = default_train_size

        train_sequences = norm_sequences[:train_size]
        test_sequences = norm_sequences[train_size:]
        train_labels = labels[:train_size]
        test_labels = labels[train_size:]

        return ExperimentDatasetClassification(
            train_sequences=train_sequences,
            train_labels=train_labels,
            test_sequences=test_sequences,
            test_labels=test_labels,
            sequence_mean=seq_mean,
            sequence_std=seq_std,
            train_size=train_size,
        )

    if dataset_name == 'mnist':
        raise ValueError(
            "Dataset 'mnist' requires training.task_type to be set to 'classification'"
        )

    if dataset_name not in data_generators:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Generate raw data

    generator_fn = data_generators[dataset_name]
    input_data, target_data = generator_fn(config.data_generation)

    # Normalize data
    normalized_input, input_mean, input_std = normalize_data(input_data)
    normalized_target, target_mean, target_std = normalize_data(target_data)

    # Split into train/test (80/20 split)
    total_length = len(normalized_input)
    target_train_fraction = getattr(config.training, "train_size", None)
    if target_train_fraction is not None:
        train_size = int(total_length * target_train_fraction)
    else:
        train_size = int(0.8 * total_length)

    train_input = normalized_input[:train_size]
    train_target = normalized_target[:train_size]
    test_input = normalized_input[train_size:]
    test_target = normalized_target[train_size:]

    return ExperimentDataset(
        train_input=train_input,
        train_target=train_target,
        test_input=test_input,
        test_target=test_target,
        target_mean=target_mean,
        target_std=target_std,
        train_size=train_size
    )
