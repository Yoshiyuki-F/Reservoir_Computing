"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/components/utils/rng.py
Random generator helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np
from jax import random


def create_jax_key(seed: Optional[int]) -> random.KeyArray:
    """Create a JAX PRNGKey from an optional seed."""
    if seed is None:
        seed = np.random.SeedSequence().entropy
    return random.PRNGKey(int(seed))


def create_numpy_rng(seed: Optional[int]) -> np.random.Generator:
    """Create a NumPy Generator from an optional seed."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))

