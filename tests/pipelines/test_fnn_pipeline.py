import jax.numpy as jnp

from reservoir.models import FlaxModelFactory


def test_run_fnn_pipeline_classification():
    cfg = {
        "type": "fnn",
        "model": {"layer_dims": [2, 4, 2]},
        "training": {"learning_rate": 1e-3, "batch_size": 2, "epochs": 2, "classification": True},
    }

    # Simple XOR-like dataset
    X = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=jnp.float64,
    )
    y = jnp.array([0, 1, 1, 0])

    model = FlaxModelFactory.create_model(cfg)
    train_metrics = model.train(X, y)
    test_metrics = model.evaluate(X, y)

    assert "final_loss" in train_metrics
    assert "accuracy" in test_metrics
    assert 0.0 <= test_metrics["accuracy"] <= 1.0
