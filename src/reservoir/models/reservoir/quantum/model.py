"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/reservoir/quantum/model.py
Quantum Reservoir Model.
Wrapper class that manages state and configuration, delegating computation to functional module.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple, Literal, Union, Optional, List
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype

from reservoir.core.identifiers import AggregationMode
from reservoir.layers.aggregation import StateAggregator
from reservoir.models.reservoir.base import Reservoir
from .backend import _ensure_tensorcircuit_initialized
from .functional import _step_jit, _forward_jit

class QuantumReservoir(Reservoir):
    """
    Quantum Reservoir Computing using TensorCircuit.
    
    - Core Logic (`_step_logic`) is separated from JIT wrappers.
    - `_forward_jit` fuses the scan loop via XLA (no nested JIT).
    - Zero-overhead recompilation (Parameters passed as arguments).
    - Strict Float32/Complex64 memory usage.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        seed: int,
        aggregation_mode: AggregationMode,
        leak_rate: float,
        measurement_basis: Literal["Z", "ZZ", "Z+ZZ"],
        encoding_strategy: Literal["Rx", "Ry", "Rz", "IQP"],
        noise_type: Literal["clean", "depolarizing", "damping"],
        noise_prob: float,
        readout_error: float,
        n_trajectories: int, # 0 means Density Matrix (default), >0 means Monte Carlo
        use_remat: bool,
        use_reuploading: bool,
        precision: Literal["complex64", "complex128"],
    ) -> None:
        """Initialize Quantum Reservoir."""
        # Ensure TC backend and patches are applied (lazy, idempotent)
        _ensure_tensorcircuit_initialized(precision)
        
        n_correlations = n_qubits * (n_qubits - 1) // 2

        # Memory Guardrail
        if n_qubits >= 20:
             import warnings
             warnings.warn(
                 f"n_qubits={n_qubits} requires >100MB per state vector (complex64). "
                 "Ensure you have sufficient VRAM.",
                 ResourceWarning
             )
        
        # --- Pre-calculate static sizes for branchless logic ---
        if measurement_basis == "Z":
            output_dim = n_qubits
            self._feedback_slice = n_qubits
            self._padding_size = 0
            
        elif measurement_basis == "ZZ":
            output_dim = n_correlations
            if n_correlations >= n_qubits:
                self._feedback_slice = n_qubits
                self._padding_size = 0
            else:
                self._feedback_slice = n_correlations
                self._padding_size = n_qubits - n_correlations
                
        elif measurement_basis == "Z+ZZ":
            output_dim = n_qubits + n_correlations
            self._feedback_slice = n_qubits
            self._padding_size = 0
        else:
            output_dim = n_qubits
            self._feedback_slice = n_qubits
            self._padding_size = 0
        
        super().__init__(n_units=output_dim, seed=seed, leak_rate=leak_rate, aggregation_mode=aggregation_mode)
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.measurement_basis = measurement_basis
        self.n_correlations = n_correlations
        self.encoding_strategy = encoding_strategy
        self.noise_type = noise_type
        self.noise_prob = float(noise_prob)
        self.readout_error = float(readout_error)
        self.n_trajectories = n_trajectories
        self.use_remat = use_remat
        self.use_reuploading = use_reuploading
        self.precision = precision

        self._rng = jax.random.key(seed)
        
        # Initialize Huge Arrays
        self._init_fixed_parameters()
        self._measurement_matrix = self._compute_measurement_matrix_vectorized()

    def _init_fixed_parameters(self) -> None:
        """Initialize fixed random parameters."""
        self.reservoir_params = jax.random.uniform(
            self._rng,
            minval=0.0,
            maxval=2 * jnp.pi,
            shape=(self.n_layers, self.n_qubits, 3)
        )
        self.initial_state_vector = jnp.zeros(self.n_qubits, dtype=jnp.float32)

    def _compute_measurement_matrix_vectorized(self) -> jnp.ndarray:
        """Vectorized Precomputation of Parity/Measurement Matrix."""
        dim = 2 ** self.n_qubits
        basis_states = jnp.arange(dim)
        
        shifts = self.n_qubits - 1 - jnp.arange(self.n_qubits)
        bits = (basis_states[:, None] >> shifts[None, :]) & 1
        z_values = (1 - 2 * bits).astype(jnp.float32)
        
        row_blocks = []
        if self.measurement_basis in ("Z", "Z+ZZ"):
            row_blocks.append(z_values.T)
        if self.measurement_basis in ("ZZ", "Z+ZZ"):
            idx_i, idx_j = jnp.triu_indices(self.n_qubits, k=1)
            zz_values = z_values[:, idx_i] * z_values[:, idx_j]
            row_blocks.append(zz_values.T)
            
            
        matrix_np = jnp.vstack(row_blocks)
        
        # Apply Readout Error (Analytical Scaling)
        # Expectation values scale by (1 - 2*epsilon)^weight
        if self.readout_error > 0.0:
            scale_factor = 1.0 - 2.0 * self.readout_error
            # Calculate Hamming weight for each row (observable)
            # Z_i has weight 1. ZZ_ij has weight 2.
            # We constructed rows explicitly:
            # First N rows are Z (weight 1)
            # Next N*(N-1)/2 rows are ZZ (weight 2)
            
            weights = []
            if self.measurement_basis in ("Z", "Z+ZZ"):
                weights.append(self._broadcast_scalar(1, self.n_qubits)) # Weight 1
            if self.measurement_basis in ("ZZ", "Z+ZZ"):
                weights.append(self._broadcast_scalar(2, self.n_correlations)) # Weight 2
                
            w_vec = jnp.concatenate(weights)
            # Reshape for broadcasting: (N_obs, 1)
            damping = (scale_factor ** w_vec)[:, None]
            matrix_np = matrix_np * damping

        return jnp.array(matrix_np)

    @staticmethod
    def _broadcast_scalar(val, count):
        return jnp.full((count,), val)

    @staticmethod
    def _prepare_input(inputs: Union[jnp.ndarray, Array]) -> Tuple[jnp.ndarray, bool]:
        """Preprocess input: cast to float32 and ensure 3D shape (Batch, Time, Feat)."""
        # Cast to float32 to ensure consistent dtypes in scan (even if x64 is enabled)
        arr = jnp.asarray(inputs, dtype=jnp.float32)
        input_was_2d = (arr.ndim == 2)
        if input_was_2d:
            arr = arr[None, :, :]
        elif arr.ndim != 3:
            raise ValueError(f"QuantumReservoir expects 2D or 3D input, got {arr.shape}")
        return arr, input_was_2d

    def initialize_state(self, batch_size: int = 1) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        state = jnp.zeros((batch_size, self.n_qubits), dtype=jnp.float32)
        if self.n_trajectories > 0:
            # Monte Carlo Mode: Return (state, key) tuple
            # Update internal RNG state to ensure fresh noise per batch
            key, self._rng = jax.random.split(self._rng)
            return state, jax.random.split(key, batch_size)
        return state

    def reset_state(self, batch_size: int) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Alias for initialize_state. Resets the reservoir to the initial ground state."""
        return self.initialize_state(batch_size)

    @jaxtyped(typechecker=beartype)
    def step(self, state: Union[Float[Array, "batch n_qubits"], Tuple[jnp.ndarray, jnp.ndarray]], input_data: Float[Array, "batch features"]) -> Tuple[Union[Float[Array, "batch n_qubits"], Tuple[jnp.ndarray, jnp.ndarray]], Float[Array, "batch output_dim"]]:
        """Batched step function for debugging/stepping."""
        # Use vmapped step logic wrapper
        step_func = partial(
            _step_jit, # Use the JIT compiled wrapper
            reservoir_params=self.reservoir_params,
            measurement_matrix=self._measurement_matrix,
            n_qubits=self.n_qubits,
            leak_rate=self.leak_rate,
            feedback_slice=self._feedback_slice,
            padding_size=self._padding_size,
            encoding_strategy=self.encoding_strategy,
            noise_type=self.noise_type,
            noise_prob=self.noise_prob,
            use_remat=self.use_remat,
            use_reuploading=self.use_reuploading
        )
        return jax.vmap(step_func, in_axes=(0, 0))(state, input_data)

    @jaxtyped(typechecker=beartype)
    def forward(self, state: Union[Float[Array, "batch n_qubits"], Tuple[jnp.ndarray, jnp.ndarray]], input_data: Float[Array, "batch time features"]) -> Tuple[Union[Float[Array, "batch n_qubits"], Tuple[jnp.ndarray, jnp.ndarray]], Float[Array, "batch time output_dim"]]:
        """Forward pass using optimized scan."""
        if input_data.ndim != 3:
            raise ValueError(f"Expected (batch, time, feat), got {input_data.shape}")

        # --- Monte Carlo Ensemble Expansion ---
        original_batch_size = input_data.shape[0]
        traj_count = self.n_trajectories if self.n_trajectories > 1 else 1
        is_ensemble = traj_count > 1
        
        run_state = state
        run_inputs = input_data
        
        if is_ensemble:
            # Polymorphic State Check
            if isinstance(state, (tuple, list)):
                state_vec, rng_keys = state
                # keys shape: (Batch, 2) -> Broaden to (Batch * K, 2)
                
                # Split each key into traj_count keys
                # vmap over batch
                keys_expanded = jax.vmap(lambda k: jax.random.split(k, traj_count))(rng_keys)
                # Shape: (Batch, K, 2) -> Flatten: (Batch*K, 2)
                keys_flat = keys_expanded.reshape(original_batch_size * traj_count, 2)
                
                # Expand Vector: (Batch, N) -> (Batch*K, N)
                state_vec_flat = jnp.repeat(state_vec, traj_count, axis=0)
                
                run_state = (state_vec_flat, keys_flat)
            else:
                # If state is array but n_trajectories > 1 (Maybe density matrix mode overridden?)
                # Just repeat state
                run_state = jnp.repeat(state, traj_count, axis=0)
                
            # Expand Inputs: (Batch, T, F) -> (Batch*K, T, F)
            run_inputs = jnp.repeat(input_data, traj_count, axis=0)
        
        # --- Execution ---
        # Transpose for scan (Time, Batch, Feat)
        inputs_time_major = jnp.swapaxes(run_inputs, 0, 1)
        
        final_state, stacked_outputs = _forward_jit(
            run_state,
            inputs_time_major,
            self.reservoir_params,
            self._measurement_matrix,
            self.n_qubits,
            self.leak_rate,
            self._feedback_slice,
            self._padding_size,
            self.encoding_strategy,
            self.noise_type,
            self.noise_prob,
            self.use_remat,
            self.use_reuploading
        )
        
        # --- Ensemble Aggregation ---
        if is_ensemble:
            # stacked_outputs: (Time, Batch*K, Out) -> Transpose -> (Batch*K, Time, Out)
            stacked_outputs = jnp.swapaxes(stacked_outputs, 0, 1) # Now (B*K, T, O)
            
            # Reshape to (Batch, K, Time, Out)
            out_shape = stacked_outputs.shape
            reshaped_out = stacked_outputs.reshape(original_batch_size, traj_count, out_shape[1], out_shape[2])
            
            # Mean over trajectories (axis 1)
            stacked_outputs = jnp.mean(reshaped_out, axis=1) # (Batch, Time, Out)
            
            # Reduce final_state (Optional implementation)
            # Just take the first trajectory's key/state to maintain shape for next steps if needed
            if isinstance(final_state, (tuple, list)):
                 f_vec, f_keys = final_state
                 # f_vec: (B*K, N) -> (B, K, N) -> Mean
                 f_vec_mean = f_vec.reshape(original_batch_size, traj_count, -1).mean(axis=1)
                 
                 # f_keys: (B*K, 2). Pick first one per batch.
                 f_keys_reshaped = f_keys.reshape(original_batch_size, traj_count, 2)
                 f_keys_reduced = f_keys_reshaped[:, 0, :]
                 
                 final_state = (f_vec_mean, f_keys_reduced)
            else:
                 final_state = final_state.reshape(original_batch_size, traj_count, -1).mean(axis=1)
            
        else:
            # Standard single trajectory / density matrix
            # Transpose back (Batch, Time, Feat)
            stacked_outputs = jnp.swapaxes(stacked_outputs, 0, 1)
        
        return final_state, stacked_outputs

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        inputs: Float[Array, "batch time features"] | Float[Array, "time features"],
        return_sequences: bool = False,
        split_name: Optional[str] = None,
        **_: Any
    ) -> Float[Array, "batch out_features"] | Float[Array, "batch time out_features"] | Float[Array, "time out_features"]:
        # Cast to float32 to ensure consistent dtypes in scan (even if x64 is enabled)
        arr, input_was_2d = self._prepare_input(inputs)
        
        batch_size = arr.shape[0]
        state = self.initialize_state(batch_size)
        _, states = self.forward(state, arr)
        
        if return_sequences:
            return states[0] if input_was_2d else states
        
        log_label = f"6:{split_name}" if split_name else None
        return self.aggregator.transform(states, log_label=log_label)



    @staticmethod
    def train(_inputs: jnp.ndarray, _targets: Any = None, **__: Any) -> Dict[str, Any]:
        # Reservoir has no trainable parameters; arguments are unused.
        return {}

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "measurement_basis": self.measurement_basis,
            "encoding_strategy": self.encoding_strategy,
            "noise_type": self.noise_type,
            "noise_prob": self.noise_prob,
            "readout_error": self.readout_error,
            "n_trajectories": self.n_trajectories,
            "use_remat": self.use_remat,
            "use_reuploading": self.use_reuploading,
            "precision": self.precision,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantumReservoir":
        try:
            return cls(
                n_qubits=int(data["n_qubits"]),
                n_layers=int(data["n_layers"]),
                seed=int(data["seed"]),
                leak_rate=float(data.get("leak_rate")),
                aggregation_mode=AggregationMode(data["aggregation"]),
                measurement_basis=data["measurement_basis"],
                encoding_strategy=data.get("encoding_strategy"),
                noise_type=data.get("noise_type", "clean"),
                noise_prob=float(data.get("noise_prob",)),
                readout_error=float(data.get("readout_error")),
                n_trajectories=int(data.get("n_trajectories")),
                use_remat=bool(data.get("use_remat")),
                use_reuploading=bool(data.get("use_reuploading")),
                precision=data.get("precision"),
            )
        except KeyError as exc:
            raise KeyError(f"Missing required quantum reservoir parameter '{exc.args[0]}'") from exc

    def get_observable_names(self) -> List[str]:
        """Generate human-readable names for the measured observables."""
        names = []
        if self.measurement_basis in ("Z", "Z+ZZ"):
            names.extend([f"Z{i}" for i in range(self.n_qubits)])
        
        if self.measurement_basis in ("ZZ", "Z+ZZ"):
            # Same order as _compute_measurement_matrix_vectorized: triu indices
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    names.append(f"Z{i}Z{j}")
        return names

    def __repr__(self) -> str:
        return (
            f"QuantumReservoir(tc_backend, n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"measurement={self.measurement_basis}, encoding={self.encoding_strategy}, "
            f"noise={self.noise_type}({self.noise_prob}), ro_err={self.readout_error}, mc={self.n_trajectories}, "
            f"remat={self.use_remat}, reup={self.use_reuploading}, {self.precision})"
        )
