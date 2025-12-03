"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/loaders.py
Dataset loader registrations."""

from __future__ import annotations

from typing import Dict, Any, Tuple

import jax.numpy as jnp

from reservoir.data.generators import generate_sine_data, generate_mnist_sequence_data, generate_mackey_glass_data
from reservoir.data.presets import get_dataset_preset
from reservoir.data.registry import DatasetRegistry


@DatasetRegistry.register("sine_wave")
def load_sine_wave(config: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load or generate sine wave data and return as (N, T, F) sequences."""
    params = dict(config.get("data_params", {}) or {})
    preset = get_dataset_preset("sine_wave")
    if preset is None:
        raise ValueError("Dataset preset 'sine_wave' is missing. Define it in reservoir.data.presets.")
    data_cfg = preset.build_config(params)
    X, y = generate_sine_data(data_cfg)
    X_arr = jnp.asarray(X, dtype=jnp.float64)
    y_arr = jnp.asarray(y, dtype=jnp.float64)

    # Ensure 3D shape (N, T, F). Treat each timestep as a length-1 sequence.
    if X_arr.ndim == 2:
        X_arr = X_arr[:, None, :]

    return X_arr, y_arr


@DatasetRegistry.register("mnist")
def load_mnist(config: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load MNIST sequence dataset as (N, T, F)."""
    if generate_mnist_sequence_data is None:
        raise ImportError("MNIST sequence loader requires torch/torchvision.")

    params = dict(config.get("data_params", {}) or {})
    preset = get_dataset_preset("mnist")
    if preset is None:
        raise ValueError("Dataset preset 'mnist' is missing. Define it in reservoir.data.presets.")
    data_cfg = preset.build_config(params)
    train_seq, train_labels = generate_mnist_sequence_data(data_cfg, split="train")
    test_seq, test_labels = generate_mnist_sequence_data(data_cfg, split="test")

    train_arr = jnp.asarray(train_seq, dtype=jnp.float64)
    test_arr = jnp.asarray(test_seq, dtype=jnp.float64)
    # Ensure (N, T, F)
    if train_arr.ndim == 2:
        train_arr = train_arr[..., None]
    if test_arr.ndim == 2:
        test_arr = test_arr[..., None]

    # Flatten any spatial dims into feature dim while preserving time length.
    train_arr = train_arr.reshape(train_arr.shape[0], train_arr.shape[1], -1)
    test_arr = test_arr.reshape(test_arr.shape[0], test_arr.shape[1], -1)

    X = jnp.concatenate([train_arr, test_arr], axis=0)
    y = jnp.concatenate([jnp.asarray(train_labels), jnp.asarray(test_labels)], axis=0)
    return X, y


@DatasetRegistry.register("mackey_glass")
def load_mackey_glass(config: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate Mackey-Glass samples (N, 1, 1) compatible with sequence models."""
    params = dict(config.get("data_params", {}) or {})
    preset = get_dataset_preset("mackey_glass")
    if preset is None:
        raise ValueError("Dataset preset 'mackey_glass' is missing. Define it in reservoir.data.presets.")
    data_cfg = preset.build_config(params)
    X, y = generate_mackey_glass_data(data_cfg)
    X_arr = jnp.asarray(X, dtype=jnp.float64)
    y_arr = jnp.asarray(y, dtype=jnp.float64)
    if X_arr.ndim == 2:
        X_arr = X_arr[:, None, :]
    return X_arr, y_arr
