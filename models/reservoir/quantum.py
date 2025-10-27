"""
Quantum Reservoir Computer implementation using PennyLane.

This module implements a quantum version of reservoir computing using
parameterized quantum circuits and variational optimization.
"""

from typing import Optional, Dict, Any, Union, Sequence

import numpy as np
import pennylane as qml

from pipelines.jax_config import ensure_x64_enabled

ensure_x64_enabled()

import jax.numpy as jnp

from .base_reservoir import BaseReservoirComputer


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

    def __init__(self, config: Union[Dict[str, Any], Any], backend: Optional[str] = 'cpu'):
        """Initialize the Quantum Reservoir Computer.

        Args:
            config: Configuration object or dictionary with quantum parameters
            backend: Classical computation backend ('cpu' or 'gpu')

        Raises:
            ValueError: If configuration parameters are invalid
        """

        super().__init__()

        # Handle both dict and object configs
        if isinstance(config, dict):
            self.config = config
            self.n_qubits = config['n_qubits']
            self.circuit_depth = config['circuit_depth']
            self.n_inputs = config['n_inputs']
            self.n_outputs = config['n_outputs']
            self.backend_type = config.get('backend', 'default.qubit')
            self.random_seed = config.get('random_seed', 42)
            self.measurement_basis = config.get('measurement_basis', 'pauli-z').lower()
            self.encoding_scheme = config.get('encoding_scheme', 'amplitude').lower()
            self.entanglement = config.get('entanglement', 'circular').lower()
            self.detuning_scale = float(config.get('detuning_scale', 1.0))
        else:
            self.config = config
            self.n_qubits = config.n_qubits
            self.circuit_depth = config.circuit_depth
            self.n_inputs = config.n_inputs
            self.n_outputs = config.n_outputs
            self.backend_type = getattr(config, 'backend', 'default.qubit')
            self.random_seed = getattr(config, 'random_seed', 42)
            self.measurement_basis = getattr(config, 'measurement_basis', 'pauli-z').lower()
            self.encoding_scheme = getattr(config, 'encoding_scheme', 'amplitude').lower()
            self.entanglement = getattr(config, 'entanglement', 'circular').lower()
            self.detuning_scale = float(getattr(config, 'detuning_scale', 1.0))

        self.backend = backend

        # Validate quantum parameters
        if self.n_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")
        if self.circuit_depth < 1:
            raise ValueError("Circuit depth must be at least 1")
        if self.n_inputs < 1:
            raise ValueError("Number of inputs must be at least 1")
        if self.measurement_basis not in {"pauli-z", "multi-pauli"}:
            raise ValueError(
                "measurement_basis must be either 'pauli-z' or 'multi-pauli'"
            )
        if self.encoding_scheme not in {"amplitude", "angle", "detuning"}:
            raise ValueError(
                "encoding_scheme must be one of {'amplitude', 'angle', 'detuning'}"
            )
        if self.entanglement not in {"circular", "full"}:
            raise ValueError("entanglement must be either 'circular' or 'full'")

        # Initialize quantum device
        self._initialize_quantum_device()

        # Initialize quantum circuit parameters
        self._initialize_quantum_params()

        # Classical output weights (trained later)
        self.W_out = None
        self.best_ridge_lambda: Optional[float] = None
        self.ridge_search_log: list[Dict[str, float]] = []
        self.last_training_mse: Optional[float] = None

    def _initialize_quantum_device(self) -> None:
        """Initialize the PennyLane quantum device."""
        # Choose backend based on preference and availability
        if self.backend_type == 'jax' and self.backend == 'gpu':
            try:
                self.device = qml.device('jax.qubit', wires=self.n_qubits)
            except:
                # Fallback to default if JAX device fails
                self.device = qml.device('default.qubit', wires=self.n_qubits)
        elif self.backend_type == 'jax':
            try:
                self.device = qml.device('jax.qubit', wires=self.n_qubits)
            except:
                self.device = qml.device('default.qubit', wires=self.n_qubits)
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

    def _measure_quantum_state(self):
        """Return measurement observables based on selected basis."""
        if self.measurement_basis == "multi-pauli":
            measurements = []
            # One-body terms
            for wire in range(self.n_qubits):
                measurements.append(qml.expval(qml.PauliX(wires=wire)))
            for wire in range(self.n_qubits):
                measurements.append(qml.expval(qml.PauliY(wires=wire)))
            for wire in range(self.n_qubits):
                measurements.append(qml.expval(qml.PauliZ(wires=wire)))

            # Two-body ZZ correlations
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    measurements.append(
                        qml.expval(qml.PauliZ(wires=i) @ qml.PauliZ(wires=j))
                    )
            return measurements

        # Default single Pauli-Z measurements
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

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
            Quantum reservoir states, shape (time_steps, n_qubits)
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
        # Validate inputs using base class
        self._validate_input_data(input_data, self.n_inputs)
        self._validate_target_data(target_data, self.n_outputs, input_data.shape[0])

        # Convert to appropriate data types
        input_data = jnp.array(input_data, dtype=jnp.float32)
        target_data = jnp.array(target_data, dtype=jnp.float32)

        # Run quantum reservoir to get quantum states
        quantum_states = self._run_quantum_reservoir(input_data)

        # Add bias term for classical readout
        bias_column = jnp.ones((quantum_states.shape[0], 1))
        X = jnp.concatenate([quantum_states, bias_column], axis=1)

        # Train classical readout layer using Ridge regression (with grid search)
        lambda_candidates = []
        if ridge_lambdas:
            lambda_candidates.extend([float(l) for l in ridge_lambdas if l is not None])
        if not lambda_candidates:
            lambda_candidates = [1e-6, 1e-5, 1e-4, 1e-3]

        # Deduplicate while preserving order
        seen = set()
        ordered_lambdas = []
        for lam in lambda_candidates:
            if lam < 0:
                continue
            if lam not in seen:
                ordered_lambdas.append(lam)
                seen.add(lam)

        XTX = X.T @ X
        XTY = X.T @ target_data

        best_lambda = None
        best_mse = float("inf")
        best_weights = None
        ridge_log: list[Dict[str, float]] = []

        identity = jnp.eye(XTX.shape[0])

        for lam in ordered_lambdas:
            lam_val = float(lam)
            A = XTX + lam_val * identity
            try:
                weights = jnp.linalg.solve(A, XTY)
            except Exception:
                weights = jnp.linalg.pinv(A) @ XTY

            preds = X @ weights
            mse = float(jnp.mean((preds - target_data) ** 2))

            ridge_log.append({"lambda": lam_val, "train_mse": mse})

            if mse < best_mse:
                best_mse = mse
                best_lambda = lam_val
                best_weights = weights

        self.W_out = best_weights
        self.best_ridge_lambda = best_lambda
        self.ridge_search_log = ridge_log
        self.last_training_mse = best_mse

        # Mark as trained
        self.trained = True

    def predict(self, input_data: jnp.ndarray) -> jnp.ndarray:
        """Generate predictions using the trained quantum reservoir.

        Args:
            input_data: Input time series, shape (time_steps, n_inputs)

        Returns:
            Predictions, shape (time_steps, n_outputs)
        """
        # Use base class validation
        super().predict(input_data)

        if self.W_out is None:
            raise ValueError("モデルが訓練されていません。先にtrain()を呼び出してください。")

        # Convert to appropriate data type
        input_data = jnp.array(input_data, dtype=jnp.float32)

        # Run quantum reservoir
        quantum_states = self._run_quantum_reservoir(input_data)

        # Add bias term
        bias_column = jnp.ones((quantum_states.shape[0], 1))
        X = jnp.concatenate([quantum_states, bias_column], axis=1)

        # Generate predictions using trained readout
        predictions = X @ self.W_out
        return predictions

    def reset_state(self) -> None:
        """Reset the quantum reservoir to initial state."""
        super().reset_state()
        self.W_out = None
        # Reinitialize quantum parameters
        self._initialize_quantum_params()
        self.best_ridge_lambda = None
        self.ridge_search_log = []
        self.last_training_mse = None

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
def create_quantum_reservoir(config: Union[Dict[str, Any], Any],
                           backend: str = 'cpu') -> QuantumReservoirComputer:
    """Create a quantum reservoir computer instance.

    Args:
        config: Configuration dictionary or object
        backend: Classical computation backend

    Returns:
        Initialized QuantumReservoirComputer instance
    """
    return QuantumReservoirComputer(config, backend)
