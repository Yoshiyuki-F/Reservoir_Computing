import jax.numpy as jnp

from reservoir.models import BaseModel
from pipelines import run_reservoir_pipeline


class DummyReservoir(BaseModel):
    """Minimal BaseModel stub to validate reservoir pipeline wiring."""

    def __init__(self):
        self.trained = False
        self.n_inputs = 1
        self.n_outputs = 1

    def train(self, X: jnp.ndarray, y: jnp.ndarray):
        self.trained = True
        return {"loss": float(jnp.mean((X - y) ** 2))}

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(X)

    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray):
        preds = self.predict(X)
        mse = jnp.mean((preds - y) ** 2)
        mae = jnp.mean(jnp.abs(preds - y))
        return {"mse": float(mse), "mae": float(mae)}


def test_run_reservoir_pipeline_with_stub_model():
    X = jnp.array([[0.0], [1.0], [2.0], [3.0]], dtype=jnp.float64)
    y = jnp.array([[0.0], [1.0], [2.0], [3.0]], dtype=jnp.float64)

    train_X, test_X = X[:3], X[3:]
    train_y, test_y = y[:3], y[3:]

    model = DummyReservoir()
    results = run_reservoir_pipeline(
        train_X=train_X,
        train_y=train_y,
        test_X=test_X,
        test_y=test_y,
        model=model,
    )

    assert model.trained is True
    assert "train" in results and "test" in results
    assert "mse" in results["test"]
    assert results["test"]["mse"] >= 0.0
