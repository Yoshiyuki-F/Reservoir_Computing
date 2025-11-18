"""Integration-style tests for pipelines.dynamic_runner wiring.

These tests focus on confirming that run_experiment wires together
model factories and regression pipelines correctly, without running
full JAX-heavy training.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax.numpy as jnp
import numpy as np

from core_lib.core import DataGenerationConfig, TrainingConfig, ModelConfig, DemoConfig, ExperimentConfig
from core_lib.data import ExperimentDataset
import pipelines.dynamic_runner as dr


class DummyReservoir:
    """Minimal stand-in for a reservoir computer.

    It records whether train/predict were called and returns simple
    deterministic outputs so that MSE computations are trivial.
    """

    def __init__(self) -> None:
        self.trained = False
        self.train_calls: list[Tuple[Any, Any, Any, Any]] = []
        self.predict_calls: list[Any] = []
        self.ridge_search_log: list[Dict[str, Any]] = []
        self.best_ridge_lambda: float | None = None

    def train(self, input_data: jnp.ndarray, target_data: jnp.ndarray, ridge_lambdas=None) -> None:  # type: ignore[override]
        self.trained = True
        self.train_calls.append((self, input_data, target_data, ridge_lambdas))

    def predict(self, input_data: jnp.ndarray) -> jnp.ndarray:  # type: ignore[override]
        self.predict_calls.append(input_data)
        # Identity mapping keeps denormalization / MSE simple
        return input_data

    def get_reservoir_info(self) -> Dict[str, Any]:  # type: ignore[override]
        return {"n_hiddenLayer": 5, "n_inputs": 1}


class DummyFactory:
    """Factory stub used to verify create_reservoir wiring."""

    def __init__(self) -> None:
        self.calls: list[Tuple[str, Any, Any]] = []

    def create_reservoir(self, reservoir_type: str, config: Any, backend: str | None = None) -> DummyReservoir:  # type: ignore[override]
        self.calls.append((reservoir_type, config, backend))
        return DummyReservoir()


def test_run_experiment_classical_regression_wiring(monkeypatch) -> None:
    """run_experiment should call classical regression pipeline with a created reservoir.

    This test avoids heavy JAX work by stubbing the model factory and
    regression train/predict functions, while still exercising the
    orchestration logic and basic metrics flow.
    """

    # Prepare a tiny fake dataset (already normalized)
    train_input = jnp.ones((4, 1), dtype=jnp.float64)
    train_target = jnp.ones((4, 1), dtype=jnp.float64)
    test_input = jnp.ones((2, 1), dtype=jnp.float64)
    test_target = jnp.ones((2, 1), dtype=jnp.float64)

    dataset = ExperimentDataset(
        train_input=train_input,
        train_target=train_target,
        test_input=test_input,
        test_target=test_target,
        target_mean=0.0,
        target_std=1.0,
        train_size=train_input.shape[0],
    )

    # Minimal experiment config for a classical reservoir regression
    data_cfg = DataGenerationConfig(
        name="sine_wave",
        time_steps=10,
        dt=1.0,
        noise_level=0.0,
        n_input=1,
        n_output=1,
        params={},
    )
    train_cfg = TrainingConfig(
        name="standard",
        task_type="timeseries",
        train_size=0.5,
        val_size=0.0,
        ridge_lambdas=[-1.0, 1.0, 3.0],
        learning_rate=0.001,
        batch_size=2,
        num_epochs=1,
    )
    model_cfg = ModelConfig(
        name="classic_reservoir",
        model_type="reservoir",
        params={"n_hiddenLayer": 5, "n_inputs": 1, "n_outputs": 1},
    )
    demo_cfg = DemoConfig(
        title="Dummy",
        filename="dummy.png",
        show_training=False,
    )
    exp_cfg = ExperimentConfig(
        data_generation=data_cfg,
        model=model_cfg,
        training=train_cfg,
        demo=demo_cfg,
        reservoir={"n_inputs": 1, "n_outputs": 1},
        quantum_reservoir={},
    )

    factory = DummyFactory()
    monkeypatch.setattr(dr, "get_model_factory", lambda model_type: factory)

    train_calls: list[Tuple[Any, Any, Any, Any]] = []
    predict_calls: list[jnp.ndarray] = []

    def stub_train(rc, inputs, targets, ridge_lambdas):
        train_calls.append((rc, inputs, targets, ridge_lambdas))

    def stub_predict(rc, inputs):
        predict_calls.append(inputs)
        return inputs

    # Override the classical regression pipeline with our stubs
    monkeypatch.setitem(dr.REGRESSION_PIPELINES, "classical", (stub_train, stub_predict))

    train_mse, test_mse, train_mae, test_mae = dr.run_experiment(
        exp_cfg,
        dataset,
        backend="cpu",
        quantum_mode=False,
        model_type="reservoir",
    )

    # Factory should have been used to construct a classical reservoir
    assert len(factory.calls) == 1
    reservoir_type, cfg_dict, backend = factory.calls[0]
    assert reservoir_type == "classical"
    assert backend == "cpu"
    assert isinstance(cfg_dict, list)

    # Regression pipeline should have been invoked with our stubs
    assert len(train_calls) == 1
    rc_arg, inputs_arg, targets_arg, lambdas_arg = train_calls[0]
    assert inputs_arg.shape == train_input.shape
    assert targets_arg.shape == train_target.shape
    assert lambdas_arg is not None

    assert len(predict_calls) >= 1
    # Basic sanity: metrics are finite numbers
    assert isinstance(test_mse, float)
    assert np.isfinite(test_mse)
    # train_mse may be None if train set is empty, but here it should be a float
    assert train_mse is None or isinstance(train_mse, float)

