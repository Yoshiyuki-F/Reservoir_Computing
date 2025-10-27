"""
MNIST dataset utilities.

Provides a simple loader that downloads (if necessary) and returns
torchvision MNIST datasets ready for CPU/GPU processing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
import numpy as np
from torchvision import datasets, transforms

MNIST_ROOT = Path("data/mnist")


def get_mnist_datasets() -> Tuple[datasets.MNIST, datasets.MNIST]:
    """
    Download (if needed) and return MNIST train/test datasets.

    Returns:
        Tuple containing torchvision MNIST training and test datasets.
    """
    transform = transforms.ToTensor()

    train_set = datasets.MNIST(
        root=str(MNIST_ROOT),
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=str(MNIST_ROOT),
        train=False,
        download=True,
        transform=transform,
    )

    return train_set, test_set


def image_to_sequence(
    image: np.ndarray,
    mode: str = "cols",
) -> np.ndarray:
    """
    Convert a single MNIST image into a 2D time-series array.

    Args:
        image: Input image array of shape (28, 28) or (1, 28, 28).
        mode: Encoding mode. \"cols\" scans column-wise (sequence length 28,
              features 28). \"flat\" flattens to 784 steps with scalar inputs.

    Returns:
        Array with shape (time_steps, features).

    Raises:
        ValueError: If an unknown mode is provided.
    """
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]

    if image.shape != (28, 28):
        raise ValueError(f"Expected image shape (28,28), got {image.shape}")

    if mode == "cols":
        sequence = np.stack([image[:, col] for col in range(28)], axis=0)
        return sequence.astype(np.float32)

    if mode == "flat":
        flat = image.reshape(-1, 1)
        return flat.astype(np.float32)

    raise ValueError("mode must be either 'cols' or 'flat'")


def get_mnist_dataloaders(
    batch_size: int = 128,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to get PyTorch DataLoaders for MNIST.

    Args:
        batch_size: Batch size for both train and test loaders.
        shuffle_train: Whether to shuffle the training dataset.
        num_workers: Number of worker processes for data loading.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    train_set, test_set = get_mnist_datasets()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
