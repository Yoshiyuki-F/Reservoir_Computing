from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from reservoir.models import BaseModel
from reservoir.pipelines.run import run_pipeline


class DummyReservoir(BaseModel):
    """Minimal BaseModel to validate DynamicRunner wiring."""

    def __init__(self):
        self.trained = False
        self.train_calls: list[Dict[str, Any]] = []

    def train(self, X: jnp.ndarray, y: jnp.ndarray):
        self.trained = True
        self.train_calls.append({"X": X, "y": y})
        return {"loss": float(jnp.mean((X - y) ** 2))}

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        return X

    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        mse = float(jnp.mean((X - y) ** 2))
        mae = float(jnp.mean(jnp.abs(X - y)))
        return {"mse": mse, "mae": mae}


def test_dynamic_runner_reservoir_wiring(monkeypatch):
    dummy_model = DummyReservoir()

    def fake_create(config, input_shape, reservoir_type=None, backend=None):
        return dummy_model

    # Patch reservoir factory
    monkeypatch.setattr("reservoir.pipelines.run.ReservoirFactory.create", fake_create)
    config = {
        "model_type": "classical_reservoir",
        "reservoir_type": "classical_reservoir",
        "reservoir_config": {"params": {"n_hidden_layer": 5}},
    }
    data = {
        "train_X": jnp.ones((4, 1)),
        "train_y": jnp.ones((4, 1)),
        "test_X": jnp.ones((2, 1)),
        "test_y": jnp.ones((2, 1)),
    }

    results = run_pipeline(config, data)

    assert dummy_model.trained is True
    assert "train" in results and "test" in results
    assert "mse" in results["test"]
    assert results["test"]["mse"] >= 0.0
