"""Data container structures for dataset splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp


@dataclass
class SplitDataset:
    """Canonical dataset split container."""

    train_X: jnp.ndarray
    train_y: jnp.ndarray
    test_X: jnp.ndarray
    test_y: jnp.ndarray
    val_X: Optional[jnp.ndarray] = None
    val_y: Optional[jnp.ndarray] = None
