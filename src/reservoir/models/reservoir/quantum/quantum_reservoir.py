"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/reservoir/quantum/quantum_reservoir.py
Quantum Reservoir Computing implementation using PennyLane.

This module implements Step 5A (Reservoir Loop) using quantum circuits:
- Encoding: Input features are encoded via Rx rotations
- Dynamics: Fixed random unitaries (CNOT ladder + random rotations)
- Measurement: Pauli-Z expectation values per qubit
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Literal
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml

from reservoir.core.identifiers import AggregationMode
from reservoir.layers.aggregation import StateAggregator
from reservoir.models.reservoir.base import Reservoir


class QuantumReservoir(Reservoir):
    """
    Quantum Reservoir Computing using PennyLane.
    
    Architecture:
        1. Encoding Layer: Rx(θ) rotations where θ = input features (scaled to [0, 2π])
        2. Reservoir Dynamics: Fixed random unitary (CNOT ladder + random rotations)
        3. Measurement: Pauli-Z expectation values on each qubit
    
    Note: All dynamics parameters are randomly initialized and FIXED (no training).
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        seed: int,
        aggregation_mode: AggregationMode,
        dynamics_type: Literal["cnot_ladder", "ising"],
        input_scaling: float,
        measurement_basis: Literal["Z", "ZZ", "Z+ZZ"],
    ) -> None:
        """
        Initialize Quantum Reservoir.
        
        Args:
            n_qubits: Number of qubits (derived from projection.n_units)
            n_layers: Number of variational layers in the reservoir dynamics
            seed: Random seed for fixed parameter initialization
            aggregation_mode: How to aggregate time steps (MEAN, LAST, SEQUENCE, etc.)
            dynamics_type: Type of reservoir dynamics ('cnot_ladder' or 'ising')
            input_scaling: Scaling factor for input features (typically 2π)
            measurement_basis: 'Z' (1st moment), 'ZZ' (correlations), or 'Z+ZZ' (both)
        """
        # Calculate output dimension based on measurement basis
        # Z: n_qubits (1st moment)
        # ZZ: n_qubits*(n_qubits-1)/2 (2-point correlations)
        # Z+ZZ: n_qubits + n_qubits*(n_qubits-1)/2 (combined)
        n_correlations = n_qubits * (n_qubits - 1) // 2
        
        if measurement_basis == "Z":
            output_dim = n_qubits
        elif measurement_basis == "ZZ":
            output_dim = n_correlations
        elif measurement_basis == "Z+ZZ":
            output_dim = n_qubits + n_correlations
        else:
            output_dim = n_qubits  # Default fallback
        
        super().__init__(n_units=output_dim, seed=seed)
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dynamics_type = dynamics_type
        self.input_scaling = float(input_scaling)
        self.measurement_basis = measurement_basis
        self.n_correlations = n_correlations
        
        if not isinstance(aggregation_mode, AggregationMode):
            raise TypeError(f"aggregation_mode must be AggregationMode, got {type(aggregation_mode)}.")
        self.aggregator = StateAggregator(mode=aggregation_mode)
        
        # Initialize random generator
        self._rng = np.random.default_rng(seed)
        
        # 2.1 Quantum device definition (PennyLane default.qubit with JAX interface)
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Initialize fixed random parameters for reservoir dynamics (2.3)
        self._init_fixed_parameters()
        
        # Create the quantum circuit (QNode)
        self._create_quantum_circuit()

    def _init_fixed_parameters(self) -> None:
        """
        Initialize fixed random parameters for the reservoir dynamics.
        These parameters are NEVER updated during training.
        """
        # Random rotation angles for each layer
        # Shape: (n_layers, n_qubits, 3) for Rx, Ry, Rz per qubit per layer
        self.reservoir_params = self._rng.uniform(
            low=0.0,
            high=2 * np.pi,
            size=(self.n_layers, self.n_qubits, 3)
        ).astype(np.float64)
        
        # Convert to JAX array (frozen)
        self.reservoir_params = jnp.array(self.reservoir_params)
        
        # For Ising dynamics: random coupling strengths
        if self.dynamics_type == "ising":
            # Coupling strengths between adjacent qubits
            self.ising_J = jnp.array(
                self._rng.uniform(-1.0, 1.0, size=self.n_qubits - 1).astype(np.float64)
            )
            # Transverse field strengths
            self.ising_h = jnp.array(
                self._rng.uniform(-1.0, 1.0, size=self.n_qubits).astype(np.float64)
            )
            # Time step for evolution
            self.ising_dt = 0.1

    def _create_quantum_circuit(self) -> None:
        """Create the PennyLane QNode for the quantum reservoir circuit."""
        
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        dynamics_type = self.dynamics_type
        input_scaling = self.input_scaling
        measurement_basis = self.measurement_basis
        
        # Store Ising params if needed
        if dynamics_type == "ising":
            ising_J = self.ising_J
            ising_h = self.ising_h
            ising_dt = self.ising_dt
        
        @qml.qnode(self.dev, interface="jax")
        def quantum_circuit(inputs: jnp.ndarray, reservoir_params: jnp.ndarray) -> list:
            """
            Quantum reservoir circuit.
            
            Args:
                inputs: Input features of shape (n_qubits,)
                reservoir_params: Fixed random parameters of shape (n_layers, n_qubits, 3)
            
            Returns:
                Expectation values based on measurement_basis setting
            """
            # 2.2 Encoding Layer: Apply Rx rotations with scaled input features
            scaled_inputs = inputs * input_scaling
            for i in range(n_qubits):
                qml.RX(scaled_inputs[i], wires=i)
            
            # 2.3 Reservoir Dynamics: Fixed random unitaries
            if dynamics_type == "cnot_ladder":
                # CNOT ladder + random single-qubit rotations
                for layer in range(n_layers):
                    # CNOT ladder for entanglement
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    
                    # Fixed random single-qubit rotations
                    for i in range(n_qubits):
                        qml.RX(reservoir_params[layer, i, 0], wires=i)
                        qml.RY(reservoir_params[layer, i, 1], wires=i)
                        qml.RZ(reservoir_params[layer, i, 2], wires=i)
                    
                    # Reverse CNOT ladder for more mixing
                    for i in range(n_qubits - 2, -1, -1):
                        qml.CNOT(wires=[i, i + 1])
            else:
                # Ising dynamics
                for _ in range(n_layers):
                    for i in range(n_qubits - 1):
                        qml.IsingZZ(ising_J[i] * ising_dt, wires=[i, i + 1])
                    for i in range(n_qubits):
                        qml.RX(2 * ising_h[i] * ising_dt, wires=i)
            
            # Measurement: Expectation values based on measurement_basis
            measurements = []
            
            # 1st moment (Z)
            if measurement_basis in ("Z", "Z+ZZ"):
                for i in range(n_qubits):
                    measurements.append(qml.expval(qml.PauliZ(i)))
                    
            # 2nd moment correlations (ZZ)
            if measurement_basis in ("ZZ", "Z+ZZ"):
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        measurements.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)))
            
            return measurements
        
        self._quantum_circuit = quantum_circuit

    def _apply_cnot_ladder_dynamics(self, params: jnp.ndarray) -> None:
        """
        Apply CNOT ladder + random single-qubit rotations.
        
        This creates entanglement between qubits and applies fixed random rotations.
        Structure per layer:
            1. CNOT cascade (0→1, 1→2, ..., n-2→n-1)
            2. Random Rx, Ry, Rz rotations on each qubit
        """
        for layer in range(self.n_layers):
            # CNOT ladder for entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Fixed random single-qubit rotations
            for i in range(self.n_qubits):
                qml.RX(params[layer, i, 0], wires=i)
                qml.RY(params[layer, i, 1], wires=i)
                qml.RZ(params[layer, i, 2], wires=i)
            
            # Reverse CNOT ladder for more mixing
            for i in range(self.n_qubits - 2, -1, -1):
                qml.CNOT(wires=[i, i + 1])

    def _apply_ising_dynamics(self) -> None:
        """
        Apply Ising Hamiltonian time evolution: e^{-iHΔt}
        
        H = -Σ J_ij Z_i Z_j - Σ h_i X_i
        
        This is approximated using Trotterization with IsingZZ and RX gates.
        """
        for _ in range(self.n_layers):
            # ZZ interactions between adjacent qubits
            for i in range(self.n_qubits - 1):
                qml.IsingZZ(self.ising_J[i] * self.ising_dt, wires=[i, i + 1])
            
            # Transverse field (X rotations)
            for i in range(self.n_qubits):
                qml.RX(2 * self.ising_h[i] * self.ising_dt, wires=i)

    def initialize_state(self, batch_size: int = 1) -> jnp.ndarray:
        """
        Initialize reservoir state.
        
        For quantum reservoir, the 'state' is used for memory/feedback and always
        matches the input dimension (n_qubits). The output dimension may differ if
        measuring multiple Pauli axes (e.g., XYZ gives 3x features).
        """
        return jnp.zeros((batch_size, self.n_qubits), dtype=jnp.float64)

    def step(self, state: jnp.ndarray, projected_input: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Single timestep of quantum reservoir processing.
        
        Args:
            state: Previous state for memory/feedback of shape (batch, n_qubits)
            projected_input: Input features of shape (batch, n_qubits)
        
        Returns:
            next_state: State for next step (n_qubits dimension, for feedback)
            output: Full measurement output (may be larger if multiple axes measured)
        """
        # Combine input with previous state for recurrence (weak feedback for memory)
        combined_input = projected_input + 0.1 * state
        
        # Process each sample in batch through quantum circuit
        # vmap for batch processing - QNode returns a tuple/list of values, stack them
        def circuit_wrapper(x):
            result = self._quantum_circuit(x, self.reservoir_params)
            # PennyLane returns list/tuple of measurements, stack into array
            return jnp.stack(result)
        
        batch_circuit = jax.vmap(circuit_wrapper)
        
        output = batch_circuit(combined_input)
        
        # For state feedback, we need a vector of size n_qubits.
        # - If "Z" or "Z+ZZ": Use the first n_qubits (which are the Z moments)
        # - If "ZZ": We don't have Z moments. Use first n_qubits of correlations (or pad if n_corr < n)
        if self.measurement_basis in ("Z", "Z+ZZ"):
            next_state = output[:, :self.n_qubits]
        else:  # "ZZ"
            if output.shape[1] >= self.n_qubits:
                next_state = output[:, :self.n_qubits]
            else:
                # Pad with zeros if we have fewer correlations than qubits (e.g. n=2 -> 1 corr)
                padding = jnp.zeros((output.shape[0], self.n_qubits - output.shape[1]))
                next_state = jnp.concatenate([output, padding], axis=1)
        
        return next_state, output

    def forward(self, state: jnp.ndarray, input_data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Process a sequence through the quantum reservoir.
        
        Args:
            state: Initial state of shape (batch, n_qubits)
            input_data: Input sequence of shape (batch, time, n_qubits)
        
        Returns:
            final_states: Final state after processing sequence
            stacked: All intermediate states of shape (batch, time, n_qubits)
        """
        if input_data.ndim != 3:
            raise ValueError(f"Expected batched sequences (batch, time, features), got {input_data.shape}")
        
        batch, time, feat = input_data.shape
        if feat != self.n_qubits:
            raise ValueError(f"Quantum reservoir expects feature dim {self.n_qubits}, got {feat}")
        
        # Transpose for scan: (time, batch, features)
        inputs_transposed = jnp.swapaxes(input_data, 0, 1)
        
        # Use jax.lax.scan for efficient sequential processing
        final_states, stacked = jax.lax.scan(self.step, state, inputs_transposed)
        
        # Transpose back: (batch, time, features)
        stacked = jnp.swapaxes(stacked, 0, 1)
        
        return final_states, stacked

    def __call__(
        self,
        inputs: jnp.ndarray,
        return_sequences: bool = False,
        split_name: str = None,
        **_: Any
    ) -> jnp.ndarray:
        """
        Process inputs through quantum reservoir and optionally aggregate.
        
        Args:
            inputs: Input data of shape (batch, time, n_qubits)
            return_sequences: If True, return full sequence; else aggregate
            split_name: Optional split name for logging
        
        Returns:
            Quantum reservoir states (aggregated if return_sequences=False)
        """
        arr = jnp.asarray(inputs, dtype=jnp.float64)
        if arr.ndim != 3:
            raise ValueError(f"QuantumReservoir expects 3D input (batch, time, features), got {arr.shape}")
        
        batch_size = arr.shape[0]
        initial_state = self.initialize_state(batch_size)
        _, states = self.forward(initial_state, arr)
        
        if return_sequences:
            return states
        return self.aggregator.transform(states)

    def get_feature_dim(self, time_steps: int) -> int:
        """Return aggregated feature dimension without running the model."""
        return self.aggregator.get_output_dim(self.n_qubits, int(time_steps))

    def train(self, inputs: jnp.ndarray, targets: Any = None, **__: Any) -> Dict[str, Any]:
        """
        Quantum Reservoir has no trainable parameters; return empty logs.
        
        Note: All parameters are fixed random values initialized at construction.
        """
        return {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize reservoir configuration."""
        data = super().to_dict()
        data.update({
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "seed": self.seed,
            "dynamics_type": self.dynamics_type,
            "input_scaling": self.input_scaling,
            "aggregation": self.aggregator.mode.value,
            "measurement_basis": self.measurement_basis,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantumReservoir":
        """Deserialize reservoir from dictionary."""
        try:
            return cls(
                n_qubits=int(data["n_qubits"]),
                n_layers=int(data["n_layers"]),
                seed=int(data["seed"]),
                dynamics_type=data["dynamics_type"],
                input_scaling=float(data["input_scaling"]),
                aggregation_mode=AggregationMode(data["aggregation"]),
                measurement_basis=data["measurement_basis"],
            )
        except KeyError as exc:
            raise KeyError(f"Missing required quantum reservoir parameter '{exc.args[0]}'") from exc

    def __repr__(self) -> str:
        return (
            f"QuantumReservoir(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"dynamics={self.dynamics_type}, measurement={self.measurement_basis})"
        )


__all__ = ["QuantumReservoir"]
