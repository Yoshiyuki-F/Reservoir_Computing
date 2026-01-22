"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/reservoir/quantum/quantum_reservoir.py
Quantum Reservoir Computing implementation using PennyLane.

This module implements Step 5A (Reservoir Loop) using quantum circuits:
- Encoding: Input features are encoded via Rx rotations
- Dynamics: Fixed random unitaries (CNOT ladder + random rotations)
- Measurement: Pauli-Z expectation values per qubit

Optimized for JAX JIT compilation with backprop diff_method.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Literal
from functools import partial
import warnings

# Suppress PennyLane FutureWarning about functools.partial
warnings.filterwarnings("ignore", message="functools.partial will be a method descriptor", category=FutureWarning)

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
        3. Measurement: Pauli-Z expectation values and/or correlations
    
    Optimization:
        - Uses 'default.qubit' with diff_method="backprop" for JAX JIT compatibility.
        - The entire step function is JIT-compiled for performance.
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
            measurement_basis: 'Z' (1st moment), 'ZZ' (2-point correlations), or 'Z+ZZ' (both)
        """
        # Calculate output dimension based on measurement basis
        n_correlations = n_qubits * (n_qubits - 1) // 2
        
        if measurement_basis == "Z":
            output_dim = n_qubits
        elif measurement_basis == "ZZ":
            output_dim = n_correlations
        elif measurement_basis == "Z+ZZ":
            output_dim = n_qubits + n_correlations
        else:
            output_dim = n_qubits
        
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
        
        # 1. Initialize fixed random parameters
        self._init_fixed_parameters()
        
        # 2. Define Quantum Device (JAX compatible)
        # Use simple default.qubit which is pure Python/JAX compatible
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # 3. Create JIT-compiled step function
        self._step_fn = self._create_jit_step_fn()

    def _init_fixed_parameters(self) -> None:
        """Initialize fixed random parameters."""
        # Random rotation angles: (n_layers, n_qubits, 3)
        self.reservoir_params = jnp.array(
            self._rng.uniform(0.0, 2 * np.pi, size=(self.n_layers, self.n_qubits, 3)).astype(np.float64)
        )
        
        # Initial state (zeros)
        self.initial_state_vector = jnp.zeros(self.n_qubits, dtype=jnp.float64)
        
        # Ising parameters if needed
        self.ising_J = None
        self.ising_h = None
        # Use a dummy array for JIT compatibility if not used (to pass as arg)
        # But clearer to just pass None or zeros if dynamics type known at compile time
        
        if self.dynamics_type == "ising":
            self.ising_J = jnp.array(
                self._rng.uniform(-1.0, 1.0, size=self.n_qubits - 1).astype(np.float64)
            )
            self.ising_h = jnp.array(
                self._rng.uniform(-1.0, 1.0, size=self.n_qubits).astype(np.float64)
            )
        else:
            # Create dummy arrays to satisfy signature if needed, though we can handle this via closure too
            # For simplicity in create_jit_step_fn, we'll use closure capture for ising params
            # or pass them. Let's pass them.
            self.ising_J = jnp.zeros((1,)) # Dummy
            self.ising_h = jnp.zeros((1,)) # Dummy

    def _create_jit_step_fn(self):
        """Creates the JIT-compiled QNode and step logic."""
        
        # Capture config variables to avoid self dependency in JIT
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        dynamics_type = self.dynamics_type
        input_scaling = self.input_scaling
        measurement_basis = self.measurement_basis
        
        # Define the QNode with backprop
        @qml.qnode(self.dev, interface="jax", diff_method="backprop")
        def circuit(inputs, params, ising_J, ising_h):
            # Encoding
            scaled_inputs = inputs * input_scaling
            for i in range(n_qubits):
                qml.RX(scaled_inputs[i], wires=i)
            
            # Dynamics
            if dynamics_type == "cnot_ladder":
                for layer in range(n_layers):
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    for i in range(n_qubits):
                        qml.RX(params[layer, i, 0], wires=i)
                        qml.RY(params[layer, i, 1], wires=i)
                        qml.RZ(params[layer, i, 2], wires=i)
                    for i in range(n_qubits - 2, -1, -1):
                        qml.CNOT(wires=[i, i + 1])
            else:
                # Ising
                dt = 0.1
                for _ in range(n_layers):
                    for i in range(n_qubits - 1):
                        qml.IsingZZ(ising_J[i] * dt, wires=[i, i + 1])
                    for i in range(n_qubits):
                        qml.RX(2 * ising_h[i] * dt, wires=i)
            
            # Measurement
            measurements = []
            # 1st moment (Z)
            if measurement_basis in ("Z", "Z+ZZ"):
                for i in range(n_qubits):
                    measurements.append(qml.expval(qml.PauliZ(i)))
            # 2nd moment (ZZ)
            if measurement_basis in ("ZZ", "Z+ZZ"):
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        measurements.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)))
            
            return measurements

        # Define the step logic
        @jax.jit
        def step_fn(state, projected_input, reservoir_params, ising_J, ising_h):
            # Recurrence: Input + Weak Feedback
            combined_input = projected_input + 0.1 * state
            
            # Execute circuit
            # PennyLane QNode output is a list/tuple of scalars for backprop
            # We stack them into a single tensor
            measurements = circuit(combined_input, reservoir_params, ising_J, ising_h)
            output = jnp.stack(measurements)
            
            # Next state (feedback) extraction
            # Always need vector of size n_qubits
            if measurement_basis in ("Z", "Z+ZZ"):
                next_state = output[:n_qubits]
            else:  # ZZ
                if output.shape[0] >= n_qubits:
                    next_state = output[:n_qubits]
                else:
                    padding = jnp.zeros((n_qubits - output.shape[0],))
                    next_state = jnp.concatenate([output, padding], axis=0)
            
            return next_state, output

        return step_fn

    def initialize_state(self, batch_size: int = 1) -> jnp.ndarray:
        """Initialize reservoir state (feedback vector)."""
        return jnp.zeros((batch_size, self.n_qubits), dtype=jnp.float64)

    def forward(self, state: jnp.ndarray, input_data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass using JIT-compiled scan."""
        if input_data.ndim != 3:
            raise ValueError(f"Expected (batch, time, feat), got {input_data.shape}")
        
        # Create a vmapped version of step_fn for the scan body
        # Takes (state(batched), input(batched)) -> (state(batched), output(batched))
        # Params are shared (unbatched)
        step_fn_vmapped = jax.vmap(
            partial(
                self._step_fn,
                reservoir_params=self.reservoir_params,
                ising_J=self.ising_J,
                ising_h=self.ising_h
            ),
            in_axes=(0, 0)
        )
        
        # Transpose to (time, batch, feat)
        inputs_transposed = jnp.swapaxes(input_data, 0, 1)
        
        # Scan
        final_state, stacked = jax.lax.scan(step_fn_vmapped, state, inputs_transposed)
        
        # Transpose back to (batch, time, feat)
        stacked = jnp.swapaxes(stacked, 0, 1)
        
        return final_state, stacked

    def __call__(
        self,
        inputs: jnp.ndarray,
        return_sequences: bool = False,
        split_name: str = None,
        **_: Any
    ) -> jnp.ndarray:
        """
        Process inputs through quantum reservoir and optionally aggregate.
        """
        arr = jnp.asarray(inputs, dtype=jnp.float64)
        if arr.ndim != 3:
            raise ValueError(f"QuantumReservoir expects 3D input, got {arr.shape}")
        
        batch_size = arr.shape[0]
        state = self.initialize_state(batch_size)
        _, states = self.forward(state, arr)
        
        if return_sequences:
            return states
        return self.aggregator.transform(states)

    def get_feature_dim(self, time_steps: int) -> int:
        """Return aggregated feature dimension without running the model."""
        return self.aggregator.get_output_dim(self.n_units, int(time_steps))

    def train(self, inputs: jnp.ndarray, targets: Any = None, **__: Any) -> Dict[str, Any]:
        """
        Quantum Reservoir has no trainable parameters; return empty logs.
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
                dynamics_type=data.get("dynamics_type", "cnot_ladder"),
                input_scaling=float(data.get("input_scaling", 2 * np.pi)),
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
