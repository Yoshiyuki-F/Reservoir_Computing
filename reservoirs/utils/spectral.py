"""Spectral-radius utilities for reservoir weight matrices."""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp


def spectral_radius_scale(
    weights,
    target_radius: float,
    *,
    iters: int = 100,
    tol: float = 1e-6,
) -> jnp.ndarray:
    """
    Scale a square matrix to reach a desired spectral radius using power iteration.

    Args:
        weights: Square matrix.
        target_radius: Desired spectral radius.
        iters: Maximum number of power iterations.
        tol: Small constant to avoid division by zero.
    """
    w = jnp.asarray(weights, dtype=jnp.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("Weights must be a square matrix.")

    vec = jnp.ones((w.shape[0],), dtype=jnp.float64)
    radius = 0.0
    for _ in range(max(1, iters)):
        vec = w @ vec
        norm = jnp.linalg.norm(vec)
        if norm < tol:
            break
        vec = vec / norm
        radius = norm

    if radius < tol:
        frob = jnp.linalg.norm(w)
        radius = frob / jnp.sqrt(w.shape[0])

    scale = target_radius / max(radius, tol)
    return w * scale
