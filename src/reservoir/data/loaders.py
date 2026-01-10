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
    generate_lorenz96_data,
)
from reservoir.data.presets import get_dataset_preset
from reservoir.data.config import (
    BaseDatasetConfig,
    SineWaveConfig,
    MackeyGlassConfig,
    LorenzConfig,
    Lorenz96Config,
    MNISTConfig,
)
from reservoir.core.presets import StrictRegistry
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
    """Load MNIST sequence dataset as canonical train/val/test splits."""
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

    # Create validation split from training data (10%)
    import numpy as np
    val_ratio = 0.1
    n_train = train_arr.shape[0]
    n_val = max(1, int(n_train * val_ratio))
    n_train_new = n_train - n_val

    val_arr = train_arr[n_train_new:]
    val_labels_arr = train_labels_arr[n_train_new:]
    train_arr = train_arr[:n_train_new]
    train_labels_arr = train_labels_arr[:n_train_new]

    return SplitDataset(
        train_X=np.asarray(train_arr),
        train_y=np.asarray(train_labels_arr),
        test_X=np.asarray(test_arr),
        test_y=np.asarray(test_labels_arr),
        val_X=np.asarray(val_arr),
        val_y=np.asarray(val_labels_arr),
    )


@register_loader(Dataset.MACKEY_GLASS)
def load_mackey_glass(config: MackeyGlassConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate Mackey-Glass samples (N, 1) compatible with sequence models."""
    import numpy as np
    
    # 1. Generate (returns jnp arrays)
    X_gen, y_gen = generate_mackey_glass_data(config)
    
    # 2. Reconstruct full sequence (N+1)
    # X_gen: (T, 1), y_gen: (T, 1)
    # y is X shifted by 1. full = [X[0], X[1]... X[T-1], y[T-1]]
    seq = np.append(np.asarray(X_gen).flatten(), np.asarray(y_gen)[-1])
    
    # 3. Downsampling (Parameterized)
    step = getattr(config, "downsample", 1)
    if step > 1:
        seq = seq[::step]
        print(f"    [Dataset] Applied downsampling: step={step}")

    # 4. Normalize (Disabled: Pipeline handles preprocessing)

    # 5. Naive MSE Baseline
    # "Naive MSE = mean((y[1:] - y[:-1])**2)"
    naive_loss = np.mean((seq[1:] - seq[:-1]) ** 2)
    print(f"    [Dataset] Naive Prediction MSE (Baseline): {naive_loss:.6f}")
    
    # 6. Re-create X, y
    X_new = seq[:-1].reshape(-1, 1)
    y_new = seq[1:].reshape(-1, 1)
    
    X_arr = jnp.asarray(X_new, dtype=jnp.float64)
    y_arr = jnp.asarray(y_new, dtype=jnp.float64)
    
    return X_arr, y_arr


@register_loader(Dataset.LORENZ)
def load_lorenz(config: LorenzConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate Lorenz attractor sequences."""
    X, y = generate_lorenz_data(config)
    X_arr = jnp.asarray(X, dtype=jnp.float64)
    y_arr = jnp.asarray(y, dtype=jnp.float64)
    # Return (T, F) so that splitting happens along the time axis (axis 0).
    # We will reshape this to (1, T, F) in load_dataset_with_validation_split.
    return X_arr, y_arr


@register_loader(Dataset.LORENZ96)
def load_lorenz96(config: Lorenz96Config) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate Lorenz 96 sequences."""
    X, y = generate_lorenz96_data(config)
    X_arr = jnp.asarray(X, dtype=jnp.float64)
    y_arr = jnp.asarray(y, dtype=jnp.float64)
    # Return (T, F) so that splitting happens along the time axis (axis 0).
    # We will reshape this to (1, T, F) in load_dataset_with_validation_split.
    # if X_arr.ndim == 2:
    #     X_arr = X_arr[:, None, :]
    # if y_arr.ndim == 2:
    #     y_arr = y_arr[:, None, :]
    return X_arr, y_arr


def load_dataset_with_validation_split(
    dataset_enum: Dataset,
    training_cfg: Optional[TrainingConfig] = None,
    require_3d: bool = True,
) -> SplitDataset:
    """
    Load dataset via registry, apply task-specific preprocessing, and split into train/val/test.
    """
    if training_cfg is None:
        training_cfg = get_training_preset("standard")

    preset = get_dataset_preset(dataset_enum)
    if preset is None:
        raise ValueError(f"Dataset preset '{dataset_enum}' is not registered.")
    loader = LOADER_REGISTRY.get(dataset_enum)
    if loader is None:
        raise ValueError(f"No loader registered for dataset '{dataset_enum}'.")

    print(f"Loading dataset: {dataset_enum.value}...")
    dataset = loader(preset.config)

    # val is needed for both tasks
    val_size = float(training_cfg.val_size)

    def _split_validation(features: jnp.ndarray, labels: jnp.ndarray) -> tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        # Always create validation split (minimum 1 sample)
        val_count = max(1, int(len(features) * val_size)) if val_size > 0 else 1
        train_count = len(features) - val_count
        if train_count < 1:
            train_count = len(features) - 1
            val_count = 1

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

        # Check if this is a chaotic dataset with LT-based splitting
        config = preset.config
        has_lt_split = (
            hasattr(config, 'train_lt') and 
            hasattr(config, 'val_lt') and 
            hasattr(config, 'test_lt') and
            hasattr(config, 'steps_per_lt')
        )

        if has_lt_split and config.steps_per_lt > 0:
            # LT-based splitting for chaotic datasets
            steps_per_lt = int(config.steps_per_lt)
            train_count = int(config.train_lt * steps_per_lt)
            val_count = int(config.val_lt * steps_per_lt)
            test_count = int(config.test_lt * steps_per_lt)
            
            required_total = train_count + val_count + test_count
            if total < required_total:
                raise ValueError(
                    f"Dataset '{dataset_enum}' has {total} samples but LT-based split requires "
                    f"{required_total} (train={train_count}, val={val_count}, test={test_count})."
                )
            
            # Use the first (train+val+test) samples, ignore any extra
            train_X, train_y = X[:train_count], y[:train_count]
            val_X, val_y = X[train_count:train_count + val_count], y[train_count:train_count + val_count]
            test_X, test_y = X[train_count + val_count:train_count + val_count + test_count], y[train_count + val_count:train_count + val_count + test_count]
            
            print(f"    [Dataset] LT-based split: train={train_count} ({config.train_lt} LT), "
                  f"val={val_count} ({config.val_lt} LT), test={test_count} ({config.test_lt} LT)")
        else:
            # Percentage-based splitting (original logic)
            train_ratio = float(training_cfg.train_size)
            test_ratio = float(training_cfg.test_ratio)

            if not (0.0 < train_ratio < 1.0):
                raise ValueError(f"train_size must be in (0,1), got {train_ratio}.")
            if not (0.0 <= test_ratio < 1.0):
                raise ValueError(f"test_ratio must be in [0,1), got {test_ratio}.")

            # Correct 8:1:1 split: calculate all counts from total
            import math
            train_count = max(1, math.ceil(total * train_ratio))
            val_count = max(1, round(total * val_size)) if val_size > 0 else 0
            test_count = total - train_count - val_count

            if train_count < 1 or test_count < 1:
                raise ValueError(f"Invalid split sizes: train={train_count}, val={val_count}, test={test_count} for total={total}.")

            train_X, train_y = X[:train_count], y[:train_count]
            # Always create validation split (minimum 1 sample)
            if val_count < 1:
                val_count = 1
                train_count = train_count - 1 if train_count > 1 else train_count
            val_X, val_y = X[train_count:train_count + val_count], y[train_count:train_count + val_count]
            test_X, test_y = X[train_count + val_count:], y[train_count + val_count:]

    # Special handling for Lorenz, Lorenz96 and Mackey-Glass
    # The user requires (Batch=1, Time, Feature) for these datasets.
    # Currently X is (Steps, Feature). We reshape to (1, Steps, Feature).
    if dataset_enum in (Dataset.LORENZ, Dataset.LORENZ96, Dataset.MACKEY_GLASS):
        def _reshape_to_batch1(arr: Optional[jnp.ndarray]) -> Optional[jnp.ndarray]:
            if arr is None: 
                return None
            # If (T, F) -> (1, T, F)
            if arr.ndim == 2:
                return arr[None, :, :]
            return arr

        train_X = _reshape_to_batch1(train_X)
        train_y = _reshape_to_batch1(train_y)
        test_X = _reshape_to_batch1(test_X)
        test_y = _reshape_to_batch1(test_y)
        val_X = _reshape_to_batch1(val_X)
        val_y = _reshape_to_batch1(val_y)

    # if require_3d:
    #     targets = {
    #         "train": train_X,
    #         "test": test_X,
    #         "val": val_X,
    #     }
    #     for split_name, arr in targets.items():
    #         if arr is None:
    #             continue
    #         if arr.ndim != 3:
    #             raise ValueError(
    #                 f"Model type '{model_type}' requires 3D input (Batch, Time, Features). "
    #                 f"Got shape {arr.shape} for split '{split_name}'. Please reshape your data source."
    #             )

    import numpy as np
    return SplitDataset(
        train_X=np.asarray(train_X),
        train_y=np.asarray(train_y),
        test_X=np.asarray(test_X),
        test_y=np.asarray(test_y),
        val_X=np.asarray(val_X),
        val_y=np.asarray(val_y),
    )
