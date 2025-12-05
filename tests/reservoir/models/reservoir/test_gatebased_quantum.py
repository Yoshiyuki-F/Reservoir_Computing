import numpy as np
import pytest
import jax.numpy as jnp

from reservoir.models.reservoir.quantum.quantum_gate_based import (
    QuantumReservoirComputer,
    compute_feature_dim,
)


np.random.seed(42)


def make_dummy_series(T: int = 120):
    x = np.sin(np.linspace(0, 8 * np.pi, T)).reshape(-1, 1)
    y = np.roll(x, -1)[:-1]
    x = x[:-1]
    return x.astype(np.float64), y.astype(np.float64)


def base_config(**overrides):
    config = {
        "n_qubits": 4,
        "circuit_depth": 4,
        "n_inputs": 1,
        "n_outputs": 1,
        "backend": "default.qubit",
        "random_seed": 42,
        "measurement_basis": "multi-pauli",
        "encoding_scheme": "detuning",
        "entanglement": "full",
        "detuning_scale": 1.0,
        "state_aggregation": "last",
        "readout_observables": ["X", "Y", "Z", "ZZ"],
    }
    config.update(overrides)
    return config


def test_initial_config_reflected():
    cfg = base_config()
    qrc = QuantumReservoirComputer(cfg)
    info = qrc.get_reservoir_info()
    circuit_info = qrc.get_quantum_circuit_info()
    assert info["n_qubits"] == 4
    assert info["circuit_depth"] == 4
    assert circuit_info["entangling_pattern"] == qrc.entanglement
    assert info["encoding_scheme"] == "detuning"
    assert info["readout_observables"] == ["X", "Y", "Z", "ZZ"]


def test_feature_dim_multi_pauli_last_and_last_mean():
    cfg = base_config()
    qrc = QuantumReservoirComputer(cfg)
    x, y = make_dummy_series(160)
    split = 120
    qrc.train(jnp.array(x[:split]), jnp.array(y[:split]))
    expected = compute_feature_dim(cfg["n_qubits"], cfg["readout_observables"], cfg["state_aggregation"])
    assert qrc.feature_dim_ == expected

    for agg in ("last_mean", "mts"):
        cfg_lm = base_config(state_aggregation=agg)
        qrc_lm = QuantumReservoirComputer(cfg_lm)
        seq = np.sin(np.linspace(0, 4 * np.pi, 60), dtype=np.float64).reshape(-1, 1)
        sequences = np.stack([seq[i : i + 20] for i in range(0, 40, 10)], axis=0)
        labels = np.arange(sequences.shape[0], dtype=np.int32) % 2
        qrc_lm.train_classification(
            jnp.array(sequences),
            jnp.array(labels),
            ridge_lambdas=[1e-3],
            num_classes=2,
        )
        expected_lm = compute_feature_dim(cfg_lm["n_qubits"], cfg_lm["readout_observables"], cfg_lm["state_aggregation"])
        assert qrc_lm.feature_dim_ == expected_lm


def test_feature_dim_z_zz_only():
    cfg = base_config(readout_observables=["Z", "ZZ"])
    qrc = QuantumReservoirComputer(cfg)
    x, y = make_dummy_series(160)
    qrc.train(jnp.array(x[:120]), jnp.array(y[:120]))
    expected = compute_feature_dim(cfg["n_qubits"], cfg["readout_observables"], cfg["state_aggregation"])
    assert qrc.feature_dim_ == expected


def test_train_and_predict_shapes():
    cfg = base_config()
    qrc = QuantumReservoirComputer(cfg)
    x, y = make_dummy_series(200)
    split = 150
    qrc.train(jnp.array(x[:split]), jnp.array(y[:split]))
    preds = qrc.predict(jnp.array(x[split:]))
    assert preds.shape == (x[split:].shape[0], 1)
    assert qrc.W_out is not None


def test_ridge_lambda_selection():
    cfg = base_config()
    qrc = QuantumReservoirComputer(cfg)
    x, y = make_dummy_series(200)
    lambdas = [1e-12, 1e-8, 1e-4, 1e-2]
    qrc.train(jnp.array(x[:200]), jnp.array(y[:200]), ridge_lambdas=lambdas)
    assert qrc.selected_lambda_ in lambdas
    best_lambda = min(qrc.train_mse_by_lambda_, key=qrc.train_mse_by_lambda_.get)
    assert qrc.selected_lambda_ == best_lambda


def test_detuning_scale_affects_mse():
    x, y = make_dummy_series(260)
    cfg_high = base_config(detuning_scale=1.0)
    cfg_low = base_config(detuning_scale=0.2)
    model_high = QuantumReservoirComputer(cfg_high)
    model_low = QuantumReservoirComputer(cfg_low)
    train_x = jnp.array(x[:200])
    train_y = jnp.array(y[:200])
    model_high.train(train_x, train_y)
    model_low.train(train_x, train_y)
    assert not np.isclose(model_high.last_training_mse, model_low.last_training_mse)


@pytest.mark.parametrize("agg", ["last_mean", "mts"])
def test_state_aggregation_last_mean_doubles_dimension(agg):
    cfg = base_config(state_aggregation=agg)
    qrc = QuantumReservoirComputer(cfg)
    seq = np.sin(np.linspace(0, 6 * np.pi, 80), dtype=np.float64).reshape(-1, 1)
    sequences = np.stack([seq[i : i + 20] for i in range(0, 40, 10)], axis=0)
    labels = np.arange(sequences.shape[0], dtype=np.int32)
    qrc.train_classification(jnp.array(sequences), jnp.array(labels), ridge_lambdas=[1e-3], num_classes=2)
    expected = compute_feature_dim(cfg["n_qubits"], cfg["readout_observables"], cfg["state_aggregation"])
    assert qrc.feature_dim_ == expected


def test_invalid_measurement_basis():
    cfg = base_config(measurement_basis="unsupported")
    with pytest.raises(NotImplementedError):
        QuantumReservoirComputer(cfg)


def test_invalid_readout_observable():
    cfg = base_config(readout_observables=["YY"])
    with pytest.raises(ValueError):
        QuantumReservoirComputer(cfg)


def test_predict_before_training_raises():
    cfg = base_config()
    qrc = QuantumReservoirComputer(cfg)
    x, _ = make_dummy_series(40)
    with pytest.raises(RuntimeError):
        qrc.predict(jnp.array(x))


def test_reproducibility_same_seed_same_results():
    cfg = base_config()
    x, y = make_dummy_series(220)
    train_x = jnp.array(x[:160])
    train_y = jnp.array(y[:160])

    model_a = QuantumReservoirComputer(cfg)
    model_a.train(train_x, train_y)
    preds_a = model_a.predict(jnp.array(x[160:]))

    model_b = QuantumReservoirComputer(cfg)
    model_b.train(train_x, train_y)
    preds_b = model_b.predict(jnp.array(x[160:]))

    assert np.allclose(np.array(model_a.W_out), np.array(model_b.W_out), atol=1e-9)
    assert np.allclose(np.array(preds_a), np.array(preds_b), atol=1e-9)


@pytest.mark.parametrize("entanglement", ["full", "ring"])
@pytest.mark.parametrize("observables", [["Z", "ZZ"], ["X", "Y", "Z", "ZZ"]])
def test_parametric_smoke_runs(entanglement, observables):
    cfg = base_config(entanglement=entanglement, readout_observables=observables)
    qrc = QuantumReservoirComputer(cfg)
    x, y = make_dummy_series(160)
    qrc.train(jnp.array(x[:120]), jnp.array(y[:120]))
    assert qrc.feature_dim_ == compute_feature_dim(cfg["n_qubits"], observables, cfg["state_aggregation"])
