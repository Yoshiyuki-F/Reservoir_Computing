import jax.numpy as jnp

from pipelines import run_fnn_pipeline
from core_lib.models import FlaxTrainingConfig
from core_lib.models.fnn import FNNModelConfig


def test_run_fnn_pipeline_classification():
    model_cfg = FNNModelConfig(layer_dims=[2, 4, 2])
    training_cfg = FlaxTrainingConfig(
        learning_rate=1e-3,
        batch_size=2,
        num_epochs=2,
        classification=True,
    )

    # Simple XOR-like dataset
    X = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    y = jnp.array([0, 1, 1, 0])

    results = run_fnn_pipeline(
        train_X=X,
        train_y=y,
        test_X=X,
        test_y=y,
        model_config=model_cfg,
        training_config=training_cfg,
    )

    assert "train" in results and "test" in results
    assert "accuracy" in results["test"]
    assert 0.0 <= results["test"]["accuracy"] <= 1.0
