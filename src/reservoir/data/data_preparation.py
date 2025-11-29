"""
src/reservoir/data/data_preparation.py
Model-agnostic splitting/normalization helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chex
import jax.numpy as jnp


@dataclass
class PreparedDataset:
    train_X: chex.Array
    train_y: chex.Array
    test_X: chex.Array
    test_y: chex.Array
    mean: Optional[float] = None
    std: Optional[float] = None


def split_and_normalize(
    inputs: chex.Array,
    targets: chex.Array,
    train_fraction: float = 0.8,
    normalize: bool = True,
) -> PreparedDataset:
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must share the first dimension")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")

    n_samples = inputs.shape[0]
    split_idx = int(n_samples * train_fraction)
    train_X, test_X = inputs[:split_idx], inputs[split_idx:]
    train_y, test_y = targets[:split_idx], targets[split_idx:]

    mean = float(jnp.mean(train_X)) if normalize else None
    std = float(jnp.std(train_X)) if normalize else None
    if normalize:
        std_safe = std if std and std > 0 else 1.0
        train_X = (train_X - mean) / std_safe
        test_X = (test_X - mean) / std_safe
        std = std_safe

    return PreparedDataset(train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y, mean=mean, std=std)
