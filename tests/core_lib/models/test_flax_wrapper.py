import jax.numpy as jnp

from core_lib.models.factories import FlaxModelFactory
from pipelines.generic_runner import UniversalPipeline


def test_fnn_classification_train_and_eval():
    config = {
        "type": "fnn",
        "model": {"layer_dims": [3, 4, 2]},
        "training": {"learning_rate": 1e-3, "batch_size": 2, "num_epochs": 2},
    }
    model = FlaxModelFactory.create_model(config)

    X = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=jnp.float32)
    y = jnp.array([0, 1, 0, 1])

    train_metrics = model.train(X, y)
    eval_metrics = model.evaluate(X, y)

    assert model.trained is True
    assert "loss" in train_metrics
    assert "accuracy" in eval_metrics
    assert 0.0 <= eval_metrics["accuracy"] <= 1.0


def test_fnn_regression_with_runner():
    config = {
        "type": "fnn",
        "model": {"layer_dims": [2, 8, 1]},
        "training": {
            "learning_rate": 1e-3,
            "batch_size": 4,
            "num_epochs": 3,
            "classification": False,
        },
    }
    model = FlaxModelFactory.create_model(config)

    X = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=jnp.float32)
    y = jnp.sum(X, axis=1, keepdims=True)

    runner = UniversalPipeline(model)
    results = runner.run(X, y, X, y)

    assert "train" in results and "test" in results
    assert "mse" in results["test"] and "mae" in results["test"]
    assert results["train"]["loss"] >= 0.0
    assert results["test"]["mse"] >= 0.0


def test_save_and_load_roundtrip(tmp_path):
    config = {
        "type": "fnn",
        "model": {"layer_dims": [2, 4, 1]},
        "training": {
            "learning_rate": 1e-3,
            "batch_size": 2,
            "num_epochs": 1,
            "classification": False,
        },
    }
    model = FlaxModelFactory.create_model(config)
    X = jnp.array([[0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32)
    y = jnp.sum(X, axis=1, keepdims=True)

    model.train(X, y)
    preds_before = model.predict(X)

    save_path = tmp_path / "params.msgpack"
    model.save(save_path)

    loaded_model = FlaxModelFactory.create_model(config)
    loaded_model.load(save_path, sample_input=X)
    preds_after = loaded_model.predict(X)

    assert jnp.allclose(preds_before, preds_after)
    assert loaded_model.trained is True
