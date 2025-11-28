import jax.numpy as jnp

from pipelines import run_rnn_pipeline
from core_lib.models import SimpleRNNConfig, FlaxTrainingConfig


def test_run_rnn_pipeline_regression():
    model_cfg = SimpleRNNConfig(
        input_dim=2,
        hidden_dim=4,
        output_dim=1,
        return_sequences=False,
    )
    training_cfg = FlaxTrainingConfig(
        learning_rate=1e-3,
        batch_size=2,
        num_epochs=2,
        classification=False,
    )

    X = jnp.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.5, 0.5], [0.5, 0.0]],
            [[1.0, 0.5], [0.5, 1.0]],
        ],
        dtype=jnp.float32,
    )
    y = jnp.sum(X, axis=(1, 2), keepdims=True)  # simple regression target

    results = run_rnn_pipeline(
        train_X=X,
        train_y=y,
        test_X=X,
        test_y=y,
        model_config=model_cfg,
        training_config=training_cfg,
    )

    assert "train" in results and "test" in results
    assert "mse" in results["test"] and "mae" in results["test"]
    assert results["test"]["mse"] >= 0.0
