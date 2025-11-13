"""State aggregation helpers shared by classical and quantum reservoirs."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

AggregationMode = Literal["last", "mean", "last_mean", "mts", "concat"]


def aggregate_states(states, mode: AggregationMode) -> jnp.ndarray:
    """
    Reduce a sequence of reservoir states into a fixed-length readout feature.

    Args:
        states: Array-like structure with shape (time_steps, feature_dim).
        mode: Aggregation strategy.

    Returns:
        Aggregated 1-D JAX array.
    """
    arr = jnp.asarray(states, dtype=jnp.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for states, got shape {arr.shape}")

    if mode == "last":
        return arr[-1]
    if mode == "mean":
        return jnp.mean(arr, axis=0)
    if mode in {"last_mean", "mts"}:
        last = arr[-1]
        mean = jnp.mean(arr, axis=0)
        return jnp.concatenate([last, mean], axis=0)
    if mode == "concat":
        return arr.reshape(-1)

    raise ValueError(f"Unsupported aggregation mode: {mode}")
