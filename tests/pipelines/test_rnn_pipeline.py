import jax.numpy as jnp

from reservoir.models import FlaxModelFactory


def test_run_rnn_pipeline_regression():
    cfg = {
        "type": "rnn",
        "model": {
            "input_dim": 2,
            "hidden_dim": 4,
            "output_dim": 1,
            "return_sequences": False,
            "return_hidden": False,
        },
        "training": {"learning_rate": 1e-3, "batch_size": 2, "epochs": 2, "classification": False},
    }

    X = jnp.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.5, 0.5], [0.5, 0.0]],
            [[1.0, 0.5], [0.5, 1.0]],
        ],
        dtype=jnp.float64,
    )
    y = jnp.sum(X, axis=(1, 2), keepdims=True)  # simple regression target

    model = FlaxModelFactory.create_model(cfg)
    train_metrics = model.train(X, y)
    test_metrics = model.evaluate(X, y)

    assert "final_loss" in train_metrics
    assert "mse" in test_metrics and "mae" in test_metrics
    assert test_metrics["mse"] >= 0.0
