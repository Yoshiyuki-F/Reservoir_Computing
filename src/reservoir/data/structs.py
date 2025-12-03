"""Data container structures for dataset splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SplitDataset:
    """Canonical dataset split container."""

    train_X: np.ndarray
    train_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    val_X: Optional[np.ndarray] = None
    val_y: Optional[np.ndarray] = None
