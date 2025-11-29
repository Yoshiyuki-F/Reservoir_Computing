"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/loaders.py
Dataset loader registrations."""

from __future__ import annotations

from typing import Dict, Any, Tuple

import jax.numpy as jnp

from reservoir.data.config import DataGenerationConfig
from reservoir.data.generators import generate_sine_data, generate_mnist_sequence_data, generate_mackey_glass_data
from reservoir.data.registry import DatasetRegistry


@DatasetRegistry.register("sine_wave")
def load_sine_wave(config: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load or generate sine wave data and return as (N, T, F) sequences."""
    params = config.get("data_params", {}) or {}
    data_cfg = DataGenerationConfig(
        name="sine_wave",
        time_steps=int(params.get("time_steps", 400)),
        dt=float(params.get("dt", 0.01)),
        noise_level=float(params.get("noise_level", 0.01)),
        params={"frequencies": params.get("frequencies", [1.0])},
    )
    X, y = generate_sine_data(data_cfg)
    X_arr = jnp.asarray(X, dtype=jnp.float32)
    y_arr = jnp.asarray(y, dtype=jnp.float32)

    # Ensure 3D shape (N, T, F). Treat each timestep as a length-1 sequence.
    if X_arr.ndim == 2:
        X_arr = X_arr[:, None, :]

    return X_arr, y_arr


@DatasetRegistry.register("mnist")
def load_mnist(config: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load MNIST sequence dataset as (N, T, F)."""
    if generate_mnist_sequence_data is None:
        raise ImportError("MNIST sequence loader requires torch/torchvision.")

    params = config.get("data_params", {}) or {}
    time_steps = int(params.get("time_steps", 28))
    n_input = int(params.get("n_input", 28))
    n_output = int(params.get("n_output", 10))
    data_cfg = DataGenerationConfig(
        name="mnist",
        time_steps=time_steps,
        dt=float(params.get("dt", 1.0)),
        noise_level=float(params.get("noise_level", 0.0)),
        n_input=n_input,
        n_output=n_output,
        params=params,
    )
    train_seq, train_labels = generate_mnist_sequence_data(data_cfg, split="train")
    test_seq, test_labels = generate_mnist_sequence_data(data_cfg, split="test")

    train_arr = jnp.asarray(train_seq, dtype=jnp.float32)
    test_arr = jnp.asarray(test_seq, dtype=jnp.float32)
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
    params = config.get("data_params", {}) or {}
    data_cfg = DataGenerationConfig(
        name="mackey_glass",
        time_steps=int(params.get("time_steps", 2000)),
        dt=float(params.get("dt", 0.1)),
        noise_level=float(params.get("noise_level", 0.0)),
        params={
            "tau": params.get("tau", 17),
            "beta": params.get("beta", 0.2),
            "gamma": params.get("gamma", 0.1),
            "n": params.get("n", 10),
            "initial_value": params.get("initial_value", 1.2),
            "warmup_steps": params.get("warmup_steps", 100),
        },
    )
    X, y = generate_mackey_glass_data(data_cfg)
    X_arr = jnp.asarray(X, dtype=jnp.float32)
    y_arr = jnp.asarray(y, dtype=jnp.float32)
    if X_arr.ndim == 2:
        X_arr = X_arr[:, None, :]
    return X_arr, y_arr
