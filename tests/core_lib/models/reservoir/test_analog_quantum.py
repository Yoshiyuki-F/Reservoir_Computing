import numpy as np
import pytest

try:
from core_lib.models.reservoir.analog_quantum import AnalogQuantumReservoir
except ImportError:  # pragma: no cover - optional dependency
    AnalogQuantumReservoir = None


pytestmark = pytest.mark.skipif(
    AnalogQuantumReservoir is None,
    reason="AnalogQuantumReservoir requires QuTiP (pip install qutip)",
)


def _simple_regression_config():
    return {
        "model_type": "analog_quantum",
        "n_qubits": 2,
        "Omega": np.pi,
        "Delta_g": 0.0,
        "Delta_l": 1.0,
        "C6": 1.0,
        "t_final": 1.0,
        "dt": 0.5,
        "encoding_scheme": "detuning",
        "measurement_basis": "multi-pauli",
        "readout_observables": ["X", "Y", "Z", "ZZ"],
        "state_aggregation": "last",
        "reupload_layers": 1,
        "input_mode": "scalar",
        "detuning_scale": 1.0,
        "ridge_lambdas": [1e-3, 1e-2],
    }


def _simple_classification_config():
    cfg = _simple_regression_config()
    cfg.update(
        {
            "reupload_layers": 2,
            "input_mode": "sequence",
            "state_aggregation": "last_mean",
        }
    )
    return cfg


def test_readout_observable_subset_controls_feature_dim():
    cfg = _simple_regression_config()
    cfg["readout_observables"] = ["Z", "ZZ"]
    cfg["state_aggregation"] = "last"
    reservoir = AnalogQuantumReservoir(cfg)

    inputs = np.array([[0.1], [0.2], [0.3]], dtype=np.float64)
    targets = np.array([[0.2], [0.3], [0.4]], dtype=np.float64)

    reservoir.train(inputs, targets)
    info = reservoir.get_reservoir_info()

    n = cfg["n_qubits"]
    expected_dim = n + (n * (n - 1)) // 2
    assert info["feature_dim"] == expected_dim
    assert info["readout_observables"] == ["Z", "ZZ"]


def test_analog_quantum_regression_roundtrip():
    cfg = _simple_regression_config()
    reservoir = AnalogQuantumReservoir(cfg)

    timesteps = 20
    x = np.linspace(0.0, 2 * np.pi, timesteps, dtype=np.float64)
    inputs = np.sin(x).reshape(-1, 1)
    targets = np.cos(x).reshape(-1, 1)

    reservoir.train(inputs, targets)
    preds = reservoir.predict(inputs[:5])

    assert preds.shape == (5, 1)
    info = reservoir.get_reservoir_info()
    assert info["feature_dim"] > 0


def test_analog_quantum_classification_pathway():
    cfg = _simple_classification_config()
    reservoir = AnalogQuantumReservoir(cfg)

    rng = np.random.default_rng(0)
    samples = 4
    sequence = rng.normal(size=(samples, 4, 4))
    labels = np.array([0, 1, 0, 1], dtype=np.int32)

    reservoir.train_classification(sequence, labels, ridge_lambdas=[1e-3])
    logits = reservoir.predict_classification(sequence, return_logits=True)
    assert logits.shape[0] == samples
    preds = reservoir.predict_classification(sequence)
    assert preds.shape == (samples,)
