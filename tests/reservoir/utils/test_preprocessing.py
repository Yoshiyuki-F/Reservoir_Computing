#!/usr/bin/env python3
"""Tests for split_and_normalize utility."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from reservoir.data.data_preparation import split_and_normalize


def test_split_shapes():
    X = jnp.arange(10.0)
    y = jnp.arange(10.0)
    prepared = split_and_normalize(X, y, train_fraction=0.6, normalize=False)
    assert prepared.train_X.shape[0] == 6
    assert prepared.test_X.shape[0] == 4
    assert prepared.train_y.shape == prepared.train_X.shape
    assert prepared.test_y.shape == prepared.test_X.shape


def test_normalization_mean_std():
    data = jnp.array([1.0, 2.0, 3.0, 4.0])
    prepared = split_and_normalize(data, data, train_fraction=1.0, normalize=True)
    assert abs(float(jnp.mean(prepared.train_X))) < 1e-10
    assert abs(float(jnp.std(prepared.train_X)) - 1.0) < 1e-10
    assert abs(prepared.mean - 2.5) < 1e-10
    assert abs(prepared.std - np.sqrt(1.25)) < 1e-10


def test_constant_input_safe_std():
    data = jnp.ones((8,))
    prepared = split_and_normalize(data, data, train_fraction=0.75)
    assert jnp.allclose(prepared.train_X, 0.0)
    assert prepared.std == 1.0


def test_invalid_fraction():
    X = jnp.arange(5.0)
    y = jnp.arange(5.0)
    with pytest.raises(ValueError):
        split_and_normalize(X, y, train_fraction=0.0)
    with pytest.raises(ValueError):
        split_and_normalize(X, y, train_fraction=1.5)


def test_mismatched_shapes():
    X = jnp.arange(5.0)
    y = jnp.arange(4.0)
    with pytest.raises(ValueError):
        split_and_normalize(X, y)
