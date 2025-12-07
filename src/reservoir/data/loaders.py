"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/loaders.py
Dataset loader registrations and preparation helpers."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from reservoir.core.identifiers import Dataset
from reservoir.data.generators import (
    generate_sine_data,
    generate_mnist_sequence_data,
    generate_mackey_glass_data,
    generate_lorenz_data,
)
from reservoir.data.presets import get_dataset_preset
from reservoir.data.config import (
    BaseDatasetConfig,
    SineWaveConfig,
    MackeyGlassConfig,
    LorenzConfig,
    MNISTConfig,
)
from reservoir.core.presets import StrictRegistry
from reservoir.models import ModelConfig
from reservoir.training.presets import TrainingConfig, get_training_preset
from reservoir.data.structs import SplitDataset


LOADER_REGISTRY: StrictRegistry[Dataset, Callable[[BaseDatasetConfig], Union[Tuple[jnp.ndarray, jnp.ndarray], SplitDataset]]] = StrictRegistry(
    {}
)


def register_loader(dataset: Dataset) -> Callable[[Callable[[BaseDatasetConfig], Any]], Callable[[BaseDatasetConfig], Any]]:
    def decorator(fn: Callable[[BaseDatasetConfig], Any]) -> Callable[[BaseDatasetConfig], Any]:
        LOADER_REGISTRY.register(dataset, fn)
        return fn

    return decorator


@register_loader(Dataset.SINE_WAVE)
def load_sine_wave(config: SineWaveConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load or generate sine wave data and return as (N, T, F) sequences."""
    X, y = generate_sine_data(config)
    X_arr = jnp.asarray(X, dtype=jnp.float64)
    y_arr = jnp.asarray(y, dtype=jnp.float64)

    # Ensure 3D shape (N, T, F). Treat each timestep as a length-1 sequence.
    if X_arr.ndim == 2:
        X_arr = X_arr[:, None, :]

    return X_arr, y_arr


@register_loader(Dataset.MNIST)
def load_mnist(config: MNISTConfig) -> SplitDataset:
    """Load MNIST sequence dataset as canonical train/test splits."""
    if generate_mnist_sequence_data is None:
        raise ImportError("MNIST sequence loader requires torch/torchvision.")
    train_seq, train_labels = generate_mnist_sequence_data(config, split=config.split)
    test_seq, test_labels = generate_mnist_sequence_data(config, split="test")

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

    num_classes = int(config.n_output)
    train_labels_arr = jax.nn.one_hot(jnp.asarray(train_labels).astype(int), num_classes)
    test_labels_arr = jax.nn.one_hot(jnp.asarray(test_labels).astype(int), num_classes)

    return SplitDataset(
        train_X=train_arr,
        train_y=train_labels_arr,
        test_X=test_arr,
        test_y=test_labels_arr,
    )


@register_loader(Dataset.MACKEY_GLASS)
def load_mackey_glass(config: MackeyGlassConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate Mackey-Glass samples (N, 1, 1) compatible with sequence models."""
    X, y = generate_mackey_glass_data(config)
    X_arr = jnp.asarray(X, dtype=jnp.float64)
    y_arr = jnp.asarray(y, dtype=jnp.float64)
    if X_arr.ndim == 2:
        X_arr = X_arr[:, None, :]
    return X_arr, y_arr


@register_loader(Dataset.LORENZ)
def load_lorenz(config: LorenzConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate Lorenz attractor sequences."""
    X, y = generate_lorenz_data(config)
    X_arr = jnp.asarray(X, dtype=jnp.float64)
    y_arr = jnp.asarray(y, dtype=jnp.float64)
    if X_arr.ndim == 2:
        X_arr = X_arr[:, None, :]
    if y_arr.ndim == 2:
        y_arr = y_arr[:, None, :]
    return X_arr, y_arr


def load_dataset_with_validation_split(
    config: ModelConfig,
    training_cfg: Optional[TrainingConfig] = None,
    *,
    model_type: str,
    require_3d: bool = True,
) -> SplitDataset:
    """
    Load dataset via registry, apply task-specific preprocessing, and split into train/val/test.
    """
    if training_cfg is None:
        training_cfg = get_training_preset("standard")

    dataset_enum = config.dataset
    preset = get_dataset_preset(dataset_enum)
    if preset is None:
        raise ValueError(f"Dataset preset '{dataset_enum}' is not registered.")
    loader = LOADER_REGISTRY.get(dataset_enum)
    if loader is None:
        raise ValueError(f"No loader registered for dataset '{dataset_enum}'.")

    print(f"Loading dataset: {dataset_enum.value}...")
    dataset = loader(preset.config)

    val_size = float(training_cfg.val_size)

    def _split_validation(features: jnp.ndarray, labels: jnp.ndarray) -> tuple[
        jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]
    ]:
        if val_size <= 0.0 or len(features) <= 1:
            return features, labels, None, None

        val_count = max(1, int(len(features) * val_size))
        train_count = len(features) - val_count
        if train_count < 1:
            train_count = len(features) - 1

        val_features = features[train_count:]
        val_labels = labels[train_count:]
        return features[:train_count], labels[:train_count], val_features, val_labels

    train_X: Union[jnp.ndarray, None]
    train_y: Union[jnp.ndarray, None]
    test_X: Union[jnp.ndarray, None]
    test_y: Union[jnp.ndarray, None]

    if isinstance(dataset, SplitDataset):
        train_X = dataset.train_X
        train_y = dataset.train_y
        test_X = dataset.test_X
        test_y = dataset.test_y

        if dataset.val_X is not None or dataset.val_y is not None:
            val_X = dataset.val_X
            val_y = dataset.val_y
        else:
            train_X, train_y, val_X, val_y = _split_validation(train_X, train_y)
    else:
        try:
            X, y = dataset
        except (TypeError, ValueError):
            raise ValueError(f"Loader for dataset '{dataset_enum}' must return (X, y) tuple or SplitDataset.")

        total = len(X)
        if total < 2:
            raise ValueError(f"Dataset '{dataset_enum}' is too small to split (size={total}).")

        train_ratio = float(training_cfg.train_size)
        test_ratio = float(training_cfg.test_ratio)

        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_size must be in (0,1), got {train_ratio}.")
        if not (0.0 <= test_ratio < 1.0):
            raise ValueError(f"test_ratio must be in [0,1), got {test_ratio}.")

        if test_ratio > 0.0:
            test_count = max(1, int(total * test_ratio))
            train_count = total - test_count
        else:
            train_count = max(1, int(total * train_ratio))
            test_count = total - train_count

        if train_count < 1 or test_count < 1:
            raise ValueError(f"Invalid split sizes: train={train_count}, test={test_count} for total={total}.")

        train_X, test_X = X[:train_count], X[train_count:]
        train_y, test_y = y[:train_count], y[train_count:]

        train_X, train_y, val_X, val_y = _split_validation(train_X, train_y)

    if require_3d:
        targets = {
            "train": train_X,
            "test": test_X,
            "val": val_X,
        }
        for split_name, arr in targets.items():
            if arr is None:
                continue
            if arr.ndim != 3:
                raise ValueError(
                    f"Model type '{model_type}' requires 3D input (Batch, Time, Features). "
                    f"Got shape {arr.shape} for split '{split_name}'. Please reshape your data source."
                )

    return SplitDataset(train_X, train_y, test_X, test_y, val_X, val_y)
