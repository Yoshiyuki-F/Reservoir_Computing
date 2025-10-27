import jax.numpy as jnp

from models.reservoir.quantum import QuantumReservoirComputer


def _base_config(**overrides):
    config = {
        "n_qubits": 3,
        "circuit_depth": 2,
        "n_inputs": 1,
        "n_outputs": 1,
        "backend": "default.qubit",
        "random_seed": 0,
        "measurement_basis": "pauli-z",
        "encoding_scheme": "angle",
    }
    config.update(overrides)
    return config


def test_pauli_z_measurement_dimension():
    cfg = _base_config()
    qrc = QuantumReservoirComputer(cfg)

    inputs = jnp.ones((5, 1), dtype=jnp.float32)
    states = qrc._run_quantum_reservoir(inputs)

    assert states.shape == (5, cfg["n_qubits"])


def test_multi_pauli_measurement_dimension():
    cfg = _base_config(measurement_basis="multi-pauli")
    qrc = QuantumReservoirComputer(cfg)

    inputs = jnp.linspace(0.0, 1.0, 5, dtype=jnp.float32).reshape(-1, 1)
    states = qrc._run_quantum_reservoir(inputs)

    n_q = cfg["n_qubits"]
    expected_features = 3 * n_q + (n_q * (n_q - 1)) // 2

    assert states.shape == (5, expected_features)


def test_full_entanglement_runs():
    cfg = _base_config(entanglement="full")
    qrc = QuantumReservoirComputer(cfg)
    inputs = jnp.ones((3, 1), dtype=jnp.float32)
    states = qrc._run_quantum_reservoir(inputs)
    assert states.shape == (3, cfg["n_qubits"])


def test_ridge_lambda_grid_search(monkeypatch):
    cfg = _base_config(measurement_basis="pauli-z")
    qrc = QuantumReservoirComputer(cfg)

    def fake_run(sequence):
        steps = sequence.shape[0]
        base = jnp.arange(steps, dtype=jnp.float32).reshape(-1, 1)
        ones = jnp.ones_like(base)
        return jnp.concatenate([base, ones], axis=1)

    monkeypatch.setattr(qrc, "_run_quantum_reservoir", fake_run)

    inputs = jnp.arange(5, dtype=jnp.float32).reshape(-1, 1)
    targets = inputs * 2.0

    qrc.train(inputs, targets, ridge_lambdas=[1e-6, 1e-3, 1e-2])

    assert qrc.W_out is not None
    assert qrc.best_ridge_lambda is not None
    assert len(qrc.ridge_search_log) >= 1
    # Ensure the best lambda is one of the provided candidates
    provided = (1e-6, 1e-3, 1e-2)
    assert any(abs(qrc.best_ridge_lambda - candidate) < 1e-12 for candidate in provided)


def test_classification_training(monkeypatch):
    cfg = _base_config(measurement_basis="pauli-z", state_aggregation="last")
    qrc = QuantumReservoirComputer(cfg)

    features = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=jnp.float64,
    )

    monkeypatch.setattr(qrc, "_encode_sequences", lambda seqs, desc=None: features)

    sequences = jnp.zeros((4, 1, 1), dtype=jnp.float64)
    labels = jnp.array([0, 1, 0, 1], dtype=jnp.int32)

    qrc.train_classification(sequences, labels, ridge_lambdas=[1e-6], num_classes=2)

    assert qrc.classification_mode
    logits = qrc.predict_classification(sequences)
    assert logits.shape == (4, 2)
