"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/structs.py
Data container structures for dataset splits."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as np


@dataclass
class SplitDataset:
    """Canonical dataset split container. Uses np.ndarray for CPU memory storage."""

    train_X: np.ndarray
    train_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
