#!/usr/bin/env python3
"""
GPU smoke test: verifies a tiny JAX computation runs on a GPU device.
Skips cleanly when no GPU is available.
"""
import pytest


def _find_gpu_devices():
    import jax
    devices = []
    try:
        devices = jax.devices()
    except Exception:
        return []
    # Match both 'gpu' and 'cuda' in device string for robustness
    return [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]


def test_gpu_smoke():
    import jax
    import jax.numpy as jnp

    gpus = _find_gpu_devices()
    if not gpus:
        pytest.skip("No GPU available; skipping GPU smoke test")

    # Run a tiny matmul on the first GPU and check result
    with jax.default_device(gpus[0]):
        x = jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3)
        w = jnp.ones((3, 1), dtype=jnp.float32)
        y = x @ w
        # Expected sums per row: [0+1+2, 3+4+5] = [3, 12]
        expected = jnp.array([[3.0], [12.0]], dtype=jnp.float32)
        assert jnp.allclose(y, expected)

