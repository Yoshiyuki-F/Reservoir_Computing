"""Model-agnostic data preparation utilities."""

from typing import Tuple, Dict, Any
from dataclasses import dataclass

import jax.numpy as jnp

from .preprocessing import normalize_data
from .generators import generate_sine_data, generate_lorenz_data, generate_mackey_glass_data


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


def prepare_experiment_data(config, quantum_mode: bool = False) -> ExperimentDataset:
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
        'mackey_glass': generate_mackey_glass_data
    }

    dataset_name = config.data_generation.name
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