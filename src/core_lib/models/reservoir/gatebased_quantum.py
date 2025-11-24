"""
Quantum Reservoir Computer implementation using PennyLane.

This module implements a quantum version of reservoir computing using
parameterized quantum circuits and variational optimization.
"""

from math import comb
from typing import Optional, Dict, Any, Sequence, Iterable, List, Tuple

import numpy as np
import pennylane as qml
import json
from pathlib import Path
from functools import lru_cache

from core_lib.utils import ensure_x64_enabled

ensure_x64_enabled()

import jax.numpy as jnp

from .base_reservoir import BaseReservoirComputer
from .config import QuantumReservoirConfig, parse_ridge_lambdas
from pipelines.gate_based_quantum_pipeline import (
    train_quantum_reservoir_regression,
    predict_quantum_reservoir_regression,
    train_quantum_reservoir_classification,
    predict_quantum_reservoir_classification,
)


@lru_cache()
def _load_shared_defaults() -> Dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "presets/models/shared_reservoir_params.json"
    data = json.loads(path.read_text())
    return dict(data.get("params", {}))


@lru_cache()
def _load_quantum_defaults() -> Dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "presets/models/gate_based_quantum.json"
    data = json.loads(path.read_text())
    params = data.get('params', {}) or {}
    base = _load_shared_defaults()
    merged = {**base, **params}
    return merged


def compute_feature_dim(
    n_qubits: int,
    observables: Sequence[str],
    aggregation: str,
) -> int:
    """Compute aggregated feature dimensionality for given observables/aggregation."""
    obs_set = {obs.upper() for obs in observables}
    one_body = sum(1 for axis in ("X", "Y", "Z") if axis in obs_set) * n_qubits
    zz_terms = comb(n_qubits, 2) if "ZZ" in obs_set else 0
    base_dim = one_body + zz_terms
    agg = aggregation.lower()
    if agg in {"last_mean", "mts"}:
        return base_dim * 2
    return base_dim


class QuantumReservoirComputer(BaseReservoirComputer):
    """Quantum Reservoir Computer using PennyLane.

    Attributes:
        n_qubits: Number of qubits in the quantum circuit
        circuit_depth: Depth of the parameterized quantum circuit
        n_inputs: Number of classical input features
        n_outputs: Number of output predictions
        backend: Quantum computing backend ('default.qubit', 'jax', etc.)
        device: PennyLane quantum device
        quantum_params: Parameters of the quantum circuit
        W_out: Classical output weights (trained via Ridge regression)

    Examples:
        Basic usage:

        >>> from configs.core import QuantumReservoirConfig
        >>> config = QuantumReservoirConfig(
        ...     n_qubits=4, circuit_depth=2,
        ...     n_inputs=1, n_outputs=1
        ... )
        >>> qrc = QuantumReservoirComputer(config)
        >>> qrc.train(input_data, target_data)
        >>> predictions = qrc.predict(test_data)
    """

    def __init__(self, config: Sequence[Dict[str, Any]], backend: Optional[str] = 'cpu'):
        """Initialize the Quantum Reservoir Computer.

        Args:
            config: Configuration object or dictionary with quantum parameters
            backend: Classical computation backend ('cpu' or 'gpu')

        Raises:
            ValueError: If configuration parameters are invalid
        """

        super().__init__()

        # Merge quantum defaults with user config
        merged: Dict[str, Any] = _load_quantum_defaults().copy()
        user_defined_readouts = False
        config_sequence: Iterable[Dict[str, Any]] = [config] if isinstance(config, dict) else config  # type: ignore[arg-type]
        for cfg in config_sequence:
            cfg_dict = dict(cfg)
            params = cfg_dict.get('params', {}) or {}
            if 'readout_observables' in cfg_dict or 'readout_observables' in params:
                user_defined_readouts = True
            merged.update({k: v for k, v in cfg_dict.items() if k not in {'name', 'description', 'params'}})
            merged.update(params)

        # Create config object - this performs all validation
        cfg = QuantumReservoirConfig(**merged)

        self.config = cfg

        # Extract validated parameters
        params = cfg.params

        self.n_qubits: int = params['n_qubits']
        self.circuit_depth: int = params['circuit_depth']
        self.n_inputs: int = params['n_inputs']
        self.n_outputs: int = params['n_outputs']
        self.backend_type: str = str(params['backend'])
        self.random_seed: int = int(params['random_seed'])
        self.measurement_basis: str = str(params['measurement_basis']).lower()
        self.encoding_scheme: str = str(params['encoding_scheme']).lower()
        self.entanglement: str = str(params['entanglement']).lower()
        self.detuning_scale: float = float(params['detuning_scale'])
        self.state_aggregation: str = str(params['state_aggregation']).lower()

        self.backend = backend

        # Handle readout_observables based on measurement_basis
        if self.measurement_basis == "multi-pauli":
            default_readouts = ["X", "Y", "Z", "ZZ"]
            allowed_readouts = {"X", "Y", "Z", "ZZ"}
        else:
            default_readouts = ["Z"]
            allowed_readouts = {"Z", "ZZ"}

        raw_readouts = params.get("readout_observables")
        if raw_readouts is None:
            candidate_readouts = list(default_readouts)
        elif isinstance(raw_readouts, (list, tuple, set)):
            candidate_readouts = [str(entry).upper() for entry in raw_readouts]
        else:
            candidate_readouts = [str(raw_readouts).upper()]

        readout_sequence: List[str] = []
        for name in candidate_readouts:
            if name not in allowed_readouts:
                continue
            if name not in readout_sequence:
                readout_sequence.append(name)

        if not readout_sequence:
            if user_defined_readouts:
                raise ValueError(
                    f"readout_observables contained no supported entries for "
                    f"measurement_basis='{self.measurement_basis}'. "
                    f"Allowed values: {sorted(allowed_readouts)}"
                )
            readout_sequence = list(default_readouts)

        self.readout_observables: Tuple[str, ...] = tuple(readout_sequence)
        self.base_feature_dim = self._compute_base_feature_dim()
        self.expected_feature_dim = compute_feature_dim(
            self.n_qubits,
            self.readout_observables,
            self.state_aggregation,
        )
        self.readout_feature_dim = self.base_feature_dim
        self.feature_dim_: Optional[int] = None

        self.selected_lambda_: Optional[float] = None
        self.train_mse_by_lambda_: Dict[float, float] = {}
        self._feature_mu_: Optional[np.ndarray] = None
        self._feature_sigma_: Optional[np.ndarray] = None
        self._feature_keep_mask_: Optional[np.ndarray] = None

        # Parse ridge_lambdas using common validation function
        self.ridge_lambdas: Sequence[float] = parse_ridge_lambdas(params)

        # Initialize quantum device
        self._initialize_quantum_device()

        # Initialize quantum circuit parameters
        self._initialize_quantum_params()

        # Classical output weights (trained later)
        self.W_out = None
        self.best_ridge_lambda: Optional[float] = None
        self.ridge_search_log: list[Dict[str, float]] = []
        self.last_training_mse: Optional[float] = None
        self.classification_mode: bool = False
        self.num_classes: Optional[int] = None

    def _initialize_quantum_device(self) -> None:
        """Initialize the PennyLane quantum device."""
        if self.backend_type == 'jax' and self.backend == 'gpu':
            self.device = qml.device('jax.qubit', wires=self.n_qubits)
        else:
            self.device = qml.device('default.qubit', wires=self.n_qubits)

    def _initialize_quantum_params(self) -> None:
        """Initialize parameters for the quantum circuit."""
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        # Parameters for parameterized quantum circuit
        # Shape: (circuit_depth, n_qubits, 3) for RY, RZ, RY rotations per layer
        self.quantum_params = np.random.uniform(
            0, 2 * np.pi,
            size=(self.circuit_depth, self.n_qubits, 3)
        )

    def _encode_input(self, classical_data: jnp.ndarray) -> None:
        """Encode classical data into quantum circuit rotations.

        Args:
            classical_data: Classical input data to encode

        Note:
            For low-dimensional inputs we use angle encoding on individual
            qubits. When the input dimensionality exceeds the number of qubits,
            we fall back to amplitude encoding with padding/truncation.
        """
        features = jnp.asarray(classical_data, dtype=jnp.float32).reshape(-1)
        n_features = features.shape[0]

        if self.encoding_scheme == "amplitude":
            normalized = features / (jnp.linalg.norm(features) + 1e-8)
            n_amplitudes = 2 ** self.n_qubits
            if normalized.shape[0] < n_amplitudes:
                padded = jnp.pad(
                    normalized,
                    (0, n_amplitudes - normalized.shape[0]),
                    mode='constant'
                )
            else:
                padded = normalized[:n_amplitudes]
            amplitudes = padded / (jnp.linalg.norm(padded) + 1e-8)
            qml.AmplitudeEmbedding(features=amplitudes, wires=range(self.n_qubits))
            return

        # Angle-based encodings (detuning currently shares implementation)
        scaled = jnp.tanh(features) * jnp.pi
        padded = jnp.zeros(self.n_qubits, dtype=jnp.float32)
        padded = padded.at[:min(n_features, self.n_qubits)].set(
            scaled[:min(n_features, self.n_qubits)]
        )
        rotation_axis = 'Z' if self.encoding_scheme == 'detuning' else 'Y'
        qml.AngleEmbedding(
            features=padded,
            wires=range(self.n_qubits),
            rotation=rotation_axis,
        )

    def _quantum_reservoir_layer(
        self,
        params: np.ndarray,
        layer_idx: int,
        drive: jnp.ndarray,
    ) -> None:
        """Apply one layer of the parameterized quantum reservoir.

        Args:
            params: Parameters for this layer, shape (n_qubits, 3)
            layer_idx: Index of the current layer
        """
        # Apply parameterized rotations
        for qubit in range(self.n_qubits):
            qml.RY(params[qubit, 0], wires=qubit)
            qml.RZ(params[qubit, 1], wires=qubit)
            qml.RY(params[qubit, 2], wires=qubit)

            if self.encoding_scheme == "detuning":
                drive_component = drive[qubit % drive.shape[0]]
                qml.RZ(self.detuning_scale * drive_component, wires=qubit)

        # Apply entangling gates based on topology
        if self.entanglement == "circular":
            for qubit in range(self.n_qubits):
                qml.CNOT(wires=[qubit, (qubit + 1) % self.n_qubits])
        else:  # full entanglement
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qml.CZ(wires=[i, j])

    def _compute_base_feature_dim(self) -> int:
        """Return the base number of measured features per time step."""
        dim = 0
        for observable in self.readout_observables:
            if observable in {"X", "Y", "Z"}:
                dim += self.n_qubits
            elif observable == "ZZ":
                dim += (self.n_qubits * (self.n_qubits - 1)) // 2
        return dim

    def _prepare_design_matrix(
        self,
        feature_matrix: jnp.ndarray,
        fit: bool = False,
    ) -> jnp.ndarray:
        features_np = np.asarray(feature_matrix, dtype=np.float64)

        if features_np.ndim != 2:
            features_np = features_np.reshape(features_np.shape[0], -1)

        if fit or self._feature_mu_ is None or self._feature_sigma_ is None:
            mu = features_np.mean(axis=0)
            sigma = features_np.std(axis=0)
            keep = sigma >= 1e-12
            if not np.any(keep):
                keep = np.ones_like(sigma, dtype=bool)
            sigma_adj = sigma.copy()
            sigma_adj[~keep] = 1.0
            self._feature_mu_ = mu
            self._feature_sigma_ = sigma_adj
            self._feature_keep_mask_ = keep

            kept_sigma = sigma_adj[keep]
            print(
                f"[gate_based-qrc] feature std range (kept) -> min={float(kept_sigma.min()):.3e}, max={float(kept_sigma.max()):.3e}"
            )
        else:
            if self._feature_mu_ is None or self._feature_sigma_ is None or self._feature_keep_mask_ is None:
                raise RuntimeError("Feature scaler has not been fitted. Call train() before predict().")
            mu = self._feature_mu_
            sigma_adj = self._feature_sigma_
            keep = self._feature_keep_mask_

        centered = (features_np - mu) / sigma_adj
        centered = centered[:, keep]

        bias = np.ones((centered.shape[0], 1), dtype=np.float64)
        design = np.concatenate([centered, bias], axis=1)
        return jnp.array(design, dtype=jnp.float64)

    def _measure_quantum_state(self):
        """Return measurement observables based on selected basis."""
        measurements: List[Any] = []

        if self.measurement_basis == "multi-pauli":
            for observable in self.readout_observables:
                if observable == "X":
                    for wire in range(self.n_qubits):
                        measurements.append(qml.expval(qml.PauliX(wires=wire)))
                elif observable == "Y":
                    for wire in range(self.n_qubits):
                        measurements.append(qml.expval(qml.PauliY(wires=wire)))
                elif observable == "Z":
                    for wire in range(self.n_qubits):
                        measurements.append(qml.expval(qml.PauliZ(wires=wire)))
                elif observable == "ZZ":
                    for i in range(self.n_qubits):
                        for j in range(i + 1, self.n_qubits):
                            measurements.append(
                                qml.expval(qml.PauliZ(wires=i) @ qml.PauliZ(wires=j))
                            )
        else:  # 'pauli-z'
            for observable in self.readout_observables:
                if observable == "Z":
                    for wire in range(self.n_qubits):
                        measurements.append(qml.expval(qml.PauliZ(wires=wire)))
                elif observable == "ZZ":
                    for i in range(self.n_qubits):
                        for j in range(i + 1, self.n_qubits):
                            measurements.append(
                                qml.expval(qml.PauliZ(wires=i) @ qml.PauliZ(wires=j))
                            )

        return measurements

    def _create_quantum_circuit(self, classical_input: jnp.ndarray) -> callable:
        """Create the quantum circuit for reservoir computation.

        Args:
            classical_input: Input data for amplitude encoding

        Returns:
            Quantum circuit function
        """
        @qml.qnode(self.device, interface='jax')
        def quantum_circuit(params):
            # 1. Encode classical input into quantum state
            self._encode_input(classical_input)
            drive = jnp.atleast_1d(classical_input)

            # 2. Apply parameterized quantum reservoir layers
            for layer_idx in range(self.circuit_depth):
                self._quantum_reservoir_layer(params[layer_idx], layer_idx, drive)

            # 3. Measure expectation values based on configuration
            return self._measure_quantum_state()

        return quantum_circuit

    def _run_quantum_reservoir(self, input_sequence: jnp.ndarray) -> jnp.ndarray:
        """Run the quantum reservoir on a sequence of inputs.

        Args:
            input_sequence: Input time series, shape (time_steps, n_inputs)

        Returns:
            Quantum reservoir features, shape (time_steps, feature_dim)
        """
        reservoir_states = []

        for t in range(input_sequence.shape[0]):
            # Get current input
            current_input = input_sequence[t]

            # Create quantum circuit for this input
            circuit = self._create_quantum_circuit(current_input)

            # Execute circuit and get measurements
            measurements = circuit(self.quantum_params)
            reservoir_states.append(measurements)

        return jnp.array(reservoir_states)

    def _aggregate_states(self, states: jnp.ndarray) -> jnp.ndarray:
        if self.state_aggregation == 'last':
            return states[-1]
        if self.state_aggregation == 'mean':
            return jnp.mean(states, axis=0)
        if self.state_aggregation in {'last_mean', 'mts'}:
            last = states[-1]
            mean = jnp.mean(states, axis=0)
            return jnp.concatenate([last, mean], axis=0)
        return states.reshape(-1)

    def _encode_sequences(self, sequences: jnp.ndarray) -> jnp.ndarray:
        encoded_states = []
        for seq in sequences:
            seq_arr = jnp.array(seq, dtype=jnp.float64)
            states = self._run_quantum_reservoir(seq_arr)
            encoded_states.append(self._aggregate_states(states))
        array = jnp.stack(encoded_states, axis=0)
        print("")
        return array

    def _fit_ridge_with_grid(
        self,
        X: jnp.ndarray,
        target_data: jnp.ndarray,
        ridge_lambdas: Optional[Sequence[float]] = None,
    ) -> None:
        X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(target_data, dtype=np.float64)

        lambdas = ridge_lambdas or self.ridge_lambdas
        lambda_candidates = [float(lam) for lam in lambdas if lam is not None and lam > 0]

        if not lambda_candidates:
            lambda_candidates = list(self.ridge_lambdas)

        lambda_candidates = sorted(set(lambda_candidates))

        n_samples = X_np.shape[0]
        if n_samples < 2:
            raise ValueError("Not enough samples to perform ridge regression")

        split_idx = int(0.9 * n_samples)
        split_idx = max(1, min(split_idx, n_samples - 1))

        X_train, X_val = X_np[:split_idx], X_np[split_idx:]
        y_train, y_val = y_np[:split_idx], y_np[split_idx:]

        feature_part = X_train[:, :-1]
        if feature_part.size:
            sigma = feature_part.std(axis=0)
            print(
                f"[gate_based-qrc] feature std range (train) -> min={float(sigma.min()):.3e}, max={float(sigma.max()):.3e}"
            )
            try:
                sv = np.linalg.svd(feature_part, compute_uv=False)
                cond_number = float(sv.max() / max(sv.min(), 1e-12))
                print(f"[gate_based-qrc] design matrix cond -> {cond_number:.3e}")
            except np.linalg.LinAlgError:
                print("[gate_based-qrc] SVD failed; skipping condition number log.")

        def ridge_via_svd(design: np.ndarray, targets: np.ndarray, lam: float) -> np.ndarray:
            U, s, Vt = np.linalg.svd(design, full_matrices=False)
            UT_y = U.T @ targets
            denom = s ** 2 + lam
            coeff = (s / denom)[:, None]
            scaled = coeff * UT_y
            return Vt.T @ scaled

        train_mse_list: List[float] = []
        val_mse_list: List[float] = []
        weights_by_lambda: Dict[float, np.ndarray] = {}

        for lam in lambda_candidates:
            weights = ridge_via_svd(X_train, y_train, lam)
            yhat_tr = X_train @ weights
            train_mse = float(np.mean((y_train - yhat_tr) ** 2))

            if X_val.size > 0:
                yhat_val = X_val @ weights
                val_mse = float(np.mean((y_val - yhat_val) ** 2))
            else:
                val_mse = train_mse

            train_mse_list.append(train_mse)
            val_mse_list.append(val_mse)
            weights_by_lambda[lam] = weights

        best_index = int(np.argmin(val_mse_list))
        best_lambda = lambda_candidates[best_index]
        best_weights_np = weights_by_lambda[best_lambda]

        print("🔍 Ridge λ grid search (VAL)")
        for lam, val_mse in zip(lambda_candidates, val_mse_list):
            print(f"  λ={lam:.2e} -> val MSE={val_mse:.6e}")

        self.W_out = jnp.array(best_weights_np, dtype=jnp.float64)
        self.best_ridge_lambda = best_lambda
        self.selected_lambda_ = best_lambda
        self.last_training_mse = float(train_mse_list[best_index])
        self.train_mse_by_lambda_ = {lam: val for lam, val in zip(lambda_candidates, val_mse_list)}
        self.ridge_search_log = [
            {
                "lambda": lam,
                "train_mse": tr,
                "val_mse": val,
            }
            for lam, tr, val in zip(lambda_candidates, train_mse_list, val_mse_list)
        ]

    def train_classification(
        self,
        sequences: jnp.ndarray,
        labels: jnp.ndarray,
        ridge_lambdas: Optional[Sequence[float]] = None,
        num_classes: int = 10,
        return_features: bool = False,
    ) -> Optional[jnp.ndarray]:
        return train_quantum_reservoir_classification(
            self,
            sequences,
            labels,
            ridge_lambdas=ridge_lambdas,
            num_classes=num_classes,
            return_features=return_features,
        )

    def predict_classification(
        self,
        sequences: Optional[jnp.ndarray] = None,
        *,
        progress_desc: Optional[str] = None,
        progress_position: int = 0,
        precomputed_features: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        return predict_quantum_reservoir_classification(
            self,
            sequences=sequences,
            progress_desc=progress_desc,
            progress_position=progress_position,
            precomputed_features=precomputed_features,
        )

    def train(
        self,
        input_data: jnp.ndarray,
        target_data: jnp.ndarray,
        ridge_lambdas: Optional[Sequence[float]] = None,
    ) -> None:
        """Train the quantum reservoir computer.

        Args:
            input_data: Input time series, shape (time_steps, n_inputs)
            target_data: Target time series, shape (time_steps, n_outputs)
            ridge_lambdas: Regularization parameter grid for ridge regression

        Note:
            This implementation uses fixed quantum parameters and only trains
            the classical readout layer. Future versions could include
            variational optimization of quantum parameters.
        """
        train_quantum_reservoir_regression(
            self,
            input_data,
            target_data,
            ridge_lambdas=ridge_lambdas,
        )

    def predict(self, input_data: jnp.ndarray) -> jnp.ndarray:
        """Generate predictions using the trained quantum reservoir."""
        return predict_quantum_reservoir_regression(self, input_data)

    def reset_state(self) -> None:
        """Reset the quantum reservoir to initial state."""
        super().reset_state()
        self.W_out = None
        # Reinitialize quantum parameters
        self._initialize_quantum_params()
        self.best_ridge_lambda = None
        self.ridge_search_log = []
        self.last_training_mse = None
        self.classification_mode = False
        self.num_classes = None
        self.selected_lambda_ = None
        self.train_mse_by_lambda_.clear()
        self.feature_dim_ = None
        self._feature_mu_ = None
        self._feature_sigma_ = None
        self._feature_keep_mask_ = None

    def get_reservoir_info(self) -> Dict[str, Any]:
        """Get information about the quantum reservoir configuration."""
        base_info = {
            "n_qubits": self.n_qubits,
            "circuit_depth": self.circuit_depth,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "backend": self.backend,
            "quantum_backend": self.backend_type,
            "device": str(self.device),
            "trained": self.trained,
            "reservoir_type": "quantum",
            "measurement_basis": self.measurement_basis,
            "encoding_scheme": self.encoding_scheme,
            "state_aggregation": self.state_aggregation,
            "readout_observables": list(self.readout_observables),
            "readout_feature_dim": self.readout_feature_dim,
            "expected_feature_dim": self.expected_feature_dim,
            "feature_dim": self.feature_dim_ if self.feature_dim_ is not None else self.expected_feature_dim,
            "selected_lambda": self.selected_lambda_,
            "train_mse_by_lambda": self.train_mse_by_lambda_,
        }

        # Add config info if available
        if hasattr(self.config, 'model_dump'):
            base_info.update(self.config.model_dump())
        elif isinstance(self.config, dict):
            base_info.update(self.config)

        return base_info

    def get_quantum_circuit_info(self) -> Dict[str, Any]:
        """Get detailed information about the quantum circuit."""
        return {
            "total_parameters": self.quantum_params.size,
            "parameter_shape": self.quantum_params.shape,
            "circuit_depth": self.circuit_depth,
            "n_qubits": self.n_qubits,
            "entangling_pattern": self.entanglement,
            "measurement_basis": self.measurement_basis,
            "encoding_scheme": self.encoding_scheme,
            "state_aggregation": self.state_aggregation,
        }

    def visualize_circuit(self, sample_input: Optional[jnp.ndarray] = None) -> str:
        """Visualize the quantum circuit structure.

        Args:
            sample_input: Sample input for circuit construction

        Returns:
            String representation of the quantum circuit
        """
        if sample_input is None:
            sample_input = jnp.ones(self.n_inputs)

        circuit = self._create_quantum_circuit(sample_input)

        # Draw the circuit
        return qml.draw(circuit)(self.quantum_params)

# Convenience function for backward compatibility
def create_quantum_reservoir(config: Dict[str, Any],
                           backend: str = 'cpu') -> QuantumReservoirComputer:
    """Create a quantum reservoir computer instance.

    Args:
        config: Configuration dictionary or object
        backend: Classical computation backend

    Returns:
        Initialized QuantumReservoirComputer instance
    """
    return QuantumReservoirComputer(config, backend)
