"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/loaders.py
Dataset loader registrations and preparation helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from reservoir.data.generators import generate_sine_data, generate_mnist_sequence_data, generate_mackey_glass_data
from reservoir.data.presets import DatasetPreset, get_dataset_preset
from reservoir.core.identifiers import Dataset
from reservoir.data.registry import DatasetRegistry
from reservoir.data.structs import SplitDataset


@DatasetRegistry.register("sine_wave")
def load_sine_wave(config: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load or generate sine wave data and return as (N, T, F) sequences."""
    params = dict(config.get("data_params", {}) or {})
    preset = get_dataset_preset(Dataset.SINE_WAVE)
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
def load_mnist(config: Dict[str, Any]) -> SplitDataset:
    """Load MNIST sequence dataset as canonical train/test splits."""
    if generate_mnist_sequence_data is None:
        raise ImportError("MNIST sequence loader requires torch/torchvision.")

    params = dict(config.get("data_params", {}) or {})
    preset = get_dataset_preset(Dataset.MNIST)
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

    train_labels_arr = jnp.asarray(train_labels)
    test_labels_arr = jnp.asarray(test_labels)

    return SplitDataset(
        train_X=train_arr,
        train_y=train_labels_arr,
        test_X=test_arr,
        test_y=test_labels_arr,
    )


@DatasetRegistry.register("mackey_glass")
def load_mackey_glass(config: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate Mackey-Glass samples (N, 1, 1) compatible with sequence models."""
    params = dict(config.get("data_params", {}) or {})
    preset = get_dataset_preset(Dataset.MACKEY_GLASS)
    if preset is None:
        raise ValueError("Dataset preset 'mackey_glass' is missing. Define it in reservoir.data.presets.")
    data_cfg = preset.build_config(params)
    X, y = generate_mackey_glass_data(data_cfg)
    X_arr = jnp.asarray(X, dtype=jnp.float64)
    y_arr = jnp.asarray(y, dtype=jnp.float64)
    if X_arr.ndim == 2:
        X_arr = X_arr[:, None, :]
    return X_arr, y_arr


def load_dataset_with_validation_split(
    config: Dict[str, Any],
    dataset_preset: DatasetPreset,
    training_cfg: Dict[str, Any],
    *,
    model_type: str,
    require_3d: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """
    Load dataset via registry, apply task-specific preprocessing, and split into train/val/test.
    """
    dataset_name = config.get("dataset")
    if isinstance(dataset_name, Dataset):
        dataset_name = dataset_name.value
    elif not isinstance(dataset_name, str):
        dataset_name = str(dataset_name)
    loader = DatasetRegistry.get(dataset_name)

    print(f"Loading dataset: {dataset_name}...")
    dataset = loader(config)

    val_size = float(training_cfg.get("val_size")) if training_cfg else 0.0

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
            raise ValueError(f"Loader for dataset '{dataset_name}' must return (X, y) tuple or SplitDataset.")

        total = len(X)
        if total < 2:
            raise ValueError(f"Dataset '{dataset_name}' is too small to split (size={total}).")

        train_ratio = float(training_cfg.get("train_size", 0.8)) if training_cfg else 0.8
        test_ratio = float(training_cfg.get("test_ratio", 0.0)) if training_cfg else 0.0

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

    # Dataset-specific preprocessing
    if dataset_name in {"mnist"}:
        num_classes = int(dataset_preset.config.n_output)
        print(f"Converting labels to one-hot vectors (classes={num_classes})...")
        train_y = jax.nn.one_hot(jnp.asarray(train_y).astype(int), num_classes)
        if val_y is not None:
            val_y = jax.nn.one_hot(jnp.asarray(val_y).astype(int), num_classes)
        test_y = jax.nn.one_hot(jnp.asarray(test_y).astype(int), num_classes)

        print("Normalizing image data to [0, 1] range (div by 255)...")
        train_X = jnp.asarray(train_X, dtype=jnp.float64) / 255.0
        if val_X is not None:
            val_X = jnp.asarray(val_X, dtype=jnp.float64) / 255.0
        if test_X is not None:
            test_X = jnp.asarray(test_X, dtype=jnp.float64) / 255.0

        # Downstream scalers should be skipped for already-normalized vision data.
        # config["use_preprocessing"] = False

    return train_X, train_y, val_X, val_y, test_X, test_y
