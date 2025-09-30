"""
Quantum Reservoir Computer implementation using PennyLane.

This module implements a quantum version of reservoir computing using
parameterized quantum circuits and variational optimization.
"""

from typing import Optional, Dict, Any, Union

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
        else:
            self.config = config
            self.n_qubits = config.n_qubits
            self.circuit_depth = config.circuit_depth
            self.n_inputs = config.n_inputs
            self.n_outputs = config.n_outputs
            self.backend_type = getattr(config, 'backend', 'default.qubit')
            self.random_seed = getattr(config, 'random_seed', 42)

        self.backend = backend

        # Validate quantum parameters
        if self.n_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")
        if self.circuit_depth < 1:
            raise ValueError("Circuit depth must be at least 1")
        if self.n_inputs < 1:
            raise ValueError("Number of inputs must be at least 1")

        # Initialize quantum device
        self._initialize_quantum_device()

        # Initialize quantum circuit parameters
        self._initialize_quantum_params()

        # Classical output weights (trained later)
        self.W_out = None

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

    def _amplitude_encoding(self, classical_data: jnp.ndarray) -> None:
        """Encode classical data into quantum amplitude encoding.

        Args:
            classical_data: Classical input data to encode

        Note:
            This is a simplified amplitude encoding. For production use,
            consider more sophisticated encoding schemes.
        """
        # Normalize data to valid amplitude range
        normalized_data = classical_data / (jnp.linalg.norm(classical_data) + 1e-8)

        # Pad or truncate to fit qubit requirements
        n_amplitudes = 2 ** self.n_qubits
        if len(normalized_data) < n_amplitudes:
            # Pad with zeros
            padded_data = jnp.pad(
                normalized_data,
                (0, n_amplitudes - len(normalized_data)),
                mode='constant'
            )
        else:
            # Truncate to fit
            padded_data = normalized_data[:n_amplitudes]

        # Normalize again to ensure unit norm
        amplitudes = padded_data / (jnp.linalg.norm(padded_data) + 1e-8)

        # Apply amplitude encoding using PennyLane
        qml.AmplitudeEmbedding(features=amplitudes, wires=range(self.n_qubits))

    def _quantum_reservoir_layer(self, params: np.ndarray, layer_idx: int) -> None:
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

        # Apply entangling gates (circular connectivity)
        for qubit in range(self.n_qubits):
            qml.CNOT(wires=[qubit, (qubit + 1) % self.n_qubits])

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
            self._amplitude_encoding(classical_input)

            # 2. Apply parameterized quantum reservoir layers
            for layer_idx in range(self.circuit_depth):
                self._quantum_reservoir_layer(params[layer_idx], layer_idx)

            # 3. Measure expectation values
            # Return expectation values of Pauli-Z for all qubits
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

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

    def train(self, input_data: jnp.ndarray, target_data: jnp.ndarray,
              reg_param: float = 1e-6) -> None:
        """Train the quantum reservoir computer.

        Args:
            input_data: Input time series, shape (time_steps, n_inputs)
            target_data: Target time series, shape (time_steps, n_outputs)
            reg_param: Regularization parameter for Ridge regression

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

        # Train classical readout layer using Ridge regression
        XTX = X.T @ X
        XTY = X.T @ target_data

        A = XTX + reg_param * jnp.eye(XTX.shape[0])

        try:
            self.W_out = jnp.linalg.solve(A, XTY)
        except:
            self.W_out = jnp.linalg.pinv(A) @ XTY

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
            "reservoir_type": "quantum"
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
            "entangling_pattern": "circular",
            "measurement_basis": "Pauli-Z",
            "encoding_scheme": "amplitude_encoding"
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
