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
    val_sequences: jnp.ndarray
    val_labels: jnp.ndarray
    test_sequences: jnp.ndarray
    test_labels: jnp.ndarray
    sequence_mean: Optional[float] = None
    sequence_std: Optional[float] = None
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0




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

        train_sequences_raw, train_labels_raw = generate_mnist_sequence_data(
            config.data_generation, split="train"
        )
        test_sequences_raw, test_labels_raw = generate_mnist_sequence_data(
            config.data_generation, split="test"
        )

        train_sequences_raw = jnp.array(train_sequences_raw, dtype=jnp.float64)
        train_labels_raw = jnp.array(train_labels_raw, dtype=jnp.int32)
        test_sequences_raw = jnp.array(test_sequences_raw, dtype=jnp.float64)
        test_labels_raw = jnp.array(test_labels_raw, dtype=jnp.int32)

        norm_train_sequences, seq_mean, seq_std = normalize_data(train_sequences_raw)
        std_safe = max(seq_std, 1e-12)
        test_sequences = (test_sequences_raw - seq_mean) / std_safe

        total_train_samples = norm_train_sequences.shape[0]
        if total_train_samples == 0:
            raise ValueError("MNIST train subset is empty; training data unavailable.")

        val_fraction = getattr(config.training, "val_size", None)
        if val_fraction is None:
            val_fraction = 0.0
        val_fraction = float(val_fraction)
        val_fraction = min(max(val_fraction, 0.0), 1.0)

        if val_fraction > 0 and total_train_samples > 1:
            val_size = int(round(total_train_samples * val_fraction))
            if val_size <= 0:
                val_size = 1
            if val_size >= total_train_samples:
                val_size = total_train_samples - 1
            train_sequences = norm_train_sequences[:-val_size]
            train_labels = train_labels_raw[:-val_size]
            val_sequences = norm_train_sequences[-val_size:]
            val_labels = train_labels_raw[-val_size:]
        else:
            val_size = 0
            train_sequences = norm_train_sequences
            train_labels = train_labels_raw
            empty_shape = (0,) + norm_train_sequences.shape[1:]
            val_sequences = jnp.empty(empty_shape, dtype=norm_train_sequences.dtype)
            val_labels = jnp.empty((0,), dtype=train_labels_raw.dtype)

        return ExperimentDatasetClassification(
            train_sequences=train_sequences,
            train_labels=train_labels,
            val_sequences=val_sequences,
            val_labels=val_labels,
            test_sequences=test_sequences,
            test_labels=test_labels_raw,
            sequence_mean=seq_mean,
            sequence_std=seq_std,
            train_size=train_sequences.shape[0],
            val_size=val_size,
            test_size=test_sequences.shape[0],
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
