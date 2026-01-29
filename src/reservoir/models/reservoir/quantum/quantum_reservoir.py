r"""
Quantum Reservoir Computing implementation using TensorCircuit.

This module implements Step 5A (Reservoir Loop) using quantum circuits:
- Encoding: Input features are encoded via Rx rotations
- Dynamics: CNOT ladder + random single-qubit rotations
- Measurement: Pauli-Z expectation values per qubit

Optimized for JAX JIT compilation with TensorCircuit backend.

Qubit Ordering: Big Endian (Qubit 0 is MSB)
-------------------------------------------
TensorCircuit follows a Big Endian convention where Qubit 0 corresponds to the 
Most Significant Bit (MSB) in the state vector index.

Memory Warning
--------------
This implementation uses Vectorized Measurement ($E = M \cdot p$), by multiplying 
a measurement matrix $(N_{obs}, 2^N)$ with the state probability vector $(2^N,)$.
- Scalable up to N=16. 
- For N=20+, matrix size ~100MB+.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple, Literal
from functools import partial



# --- Lazy Initialization for Safety & Isolation ---
_TC_INITIALIZED = False

def _ensure_tensorcircuit_initialized():
    """
    Lazily configure TensorCircuit and patch Numpy.
    Ensures global side effects only happen when QuantumReservoir is verified to be used.
    """
    global _TC_INITIALIZED
    if _TC_INITIALIZED:
        return

    # Configure TensorCircuit
    # Localize backend setting to avoid affecting other modules on import
    tc.set_backend("jax")
    tc.set_dtype("complex64") # Optim: Use complex64 as per Mixed Precision strategy

    # --- Numpy 2.x Compatibility Patch ---
    # TensorCircuit usage of 'newshape' argument is incompatible with some Numpy versions/wrappers.
    # We patch it globally ONLY when this class is instantiated.
    if not hasattr(jnp, "ComplexWarning"):
        jnp.ComplexWarning = UserWarning

    # Check if already patched to prevent recursion
    if getattr(jnp.reshape, "__name__", "") != "_patched_reshape":
        _orig_reshape = jnp.reshape
        
        def _patched_reshape(a, *args, **kwargs):
            if 'newshape' in kwargs:
                args = args + (kwargs.pop('newshape'),)
            return _orig_reshape(a, *args, **kwargs)
            
        jnp.reshape = _patched_reshape

    _TC_INITIALIZED = True

import jax
import jax.numpy as jnp
import tensorcircuit as tc
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype

from reservoir.core.identifiers import AggregationMode
from reservoir.layers.aggregation import StateAggregator
from reservoir.models.reservoir.base import Reservoir


# --- Pure Logic Functions (No JIT Decoration) ---

def _make_circuit_logic(
    inputs: jnp.ndarray,
    params: jnp.ndarray,
    n_qubits: int,
    input_scaling: float,
    encoding_strategy: str,
    noise_type: str,
    noise_prob: float
) -> jnp.ndarray:
    """
    Core circuit construction logic with Noise and Encoding support.
    Refactored to use `jax.lax.scan` for the layer loop.
    Compilation time is O(1) with respect to depth.
    """
    is_noisy = (noise_type != "clean")

    # --- 1. Encoding (Input -> State) ---
    c_enc = tc.Circuit(n_qubits)
    scaled_inputs = inputs * input_scaling
    
    for i in range(n_qubits):
        if encoding_strategy == "Rx":
            c_enc.rx(i, theta=scaled_inputs[i])
        elif encoding_strategy == "Ry":
            c_enc.ry(i, theta=scaled_inputs[i])
        elif encoding_strategy == "Rz":
            c_enc.rz(i, theta=scaled_inputs[i])
        elif encoding_strategy == "IQP":
            # Simple IQP-like encoding: H -> Rz(x)
            c_enc.h(i)
            c_enc.rz(i, theta=scaled_inputs[i])
            
    # Initial State
    # If noisy, state might implicitly be DM if channels were added?
    # Actually, encoding is usually noise-free in this model, but let's check.
    # We will assume Encoding is ideal for now, or apply noise if configured?
    # User said "Inject noise after each gate". For simplicity, we apply noise in Dynamics layers.
    state = c_enc.state()

    # scanに入力する形状を確定させる
    if is_noisy:
        if state.ndim == 1:
            state = jnp.outer(state, jnp.conj(state))
        # ここで確実に (2^N, 2^N) であることを保証

    # --- 2. Dynamics (Layer Scanned) ---
    def layer_step(carry_state, layer_params):
        if is_noisy:
            # inputsが密度行列であることを明示、またはDMCircuitを使用
            c = tc.DMCircuit(n_qubits, inputs=carry_state)
        else:
            c = tc.Circuit(n_qubits, inputs=carry_state)
        
        # Apply Entanglement (CNOT Ladder)
        for j in range(n_qubits - 1):
            c.cnot(j, j + 1)
            if is_noisy and noise_type == "depolarizing":
                c.depolarizing(j, px=noise_prob, py=noise_prob, pz=noise_prob)
                c.depolarizing(j+1, px=noise_prob, py=noise_prob, pz=noise_prob)
            elif is_noisy and noise_type == "damping":
                c.amplitude_damping(j, gamma=noise_prob)
                c.amplitude_damping(j+1, gamma=noise_prob)
            
        # Apply Rotations
        for k in range(n_qubits):
            c.rx(k, theta=layer_params[k, 0])
            if is_noisy and noise_type == "depolarizing": c.depolarizing(k, px=noise_prob, py=noise_prob, pz=noise_prob)
            elif is_noisy and noise_type == "damping": c.amplitude_damping(k, gamma=noise_prob)
                
            c.ry(k, theta=layer_params[k, 1])
            if is_noisy and noise_type == "depolarizing": c.depolarizing(k, px=noise_prob, py=noise_prob, pz=noise_prob)
            elif is_noisy and noise_type == "damping": c.amplitude_damping(k, gamma=noise_prob)
                
            c.rz(k, theta=layer_params[k, 2])
            if is_noisy and noise_type == "depolarizing": c.depolarizing(k, px=noise_prob, py=noise_prob, pz=noise_prob)
            elif is_noisy and noise_type == "damping": c.amplitude_damping(k, gamma=noise_prob)
            
        # Apply Entanglement (Reverse Ladder)
        for l in range(n_qubits - 2, -1, -1):
            c.cnot(l, l + 1)
            if is_noisy and noise_type == "depolarizing":
                c.depolarizing(l, px=noise_prob, py=noise_prob, pz=noise_prob)
                c.depolarizing(l+1, px=noise_prob, py=noise_prob, pz=noise_prob)
            elif is_noisy and noise_type == "damping":
                c.amplitude_damping(l, gamma=noise_prob)
                c.amplitude_damping(l+1, gamma=noise_prob)
            
        new_state = c.state()

        # JITコンパイル時の形状チェックを安定させる
        if is_noisy:
            # もし何らかの理由でベクトルが返ってきた場合、DMへ昇格させる
            # (reshapeだけだと、純粋状態に戻ってしまった場合に計算がおかしくなる)
            if new_state.ndim == 1:
                new_state = jnp.outer(new_state, jnp.conj(new_state))
            elif new_state.ndim == 2:
                pass  # OK
            else:
                # FlattenされたDMの場合の救済
                new_state = new_state.reshape(carry_state.shape)

        return new_state, None

    # Scan over layers
    final_state, _ = jax.lax.scan(layer_step, state, params)
    
    # --- 3. Measurement ---
    if is_noisy:
        # final_state is Density Matrix (2^N, 2^N)
        # Probabilities are diagonal elements.
        probs = jnp.real(jnp.diag(final_state))
        
        # Guard: Ensure shape is (2^N,)
        # In JAX/TC, diag might keep complex type, but imaginary part should be 0 for valid DM.
        return probs.astype(jnp.float32)
    else:
        # final_state is Vector (2^N,)
        return (jnp.abs(final_state) ** 2).astype(jnp.float32)


def _step_logic(
    state: jnp.ndarray,
    input_slice: jnp.ndarray,
    reservoir_params: jnp.ndarray,
    measurement_matrix: jnp.ndarray,
    n_qubits: int,
    input_scaling: float,
    feedback_scale: float,
    feedback_slice: int,
    padding_size: int,
    encoding_strategy: str,
    noise_type: str,
    noise_prob: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Core step logic (Single step).
    NOT JIT compiled here. Used by both standalone JIT and Scan JIT functions.
    """
    # Feedback logic
    combined_input = input_slice + feedback_scale * state
    
    # Circuit Execution
    probs = _make_circuit_logic(
        combined_input, 
        reservoir_params, 
        n_qubits, 
        input_scaling,
        encoding_strategy,
        noise_type,
        noise_prob
    )
    
    # Vectorized Measurement: E = M @ p
    output = jnp.dot(measurement_matrix, probs)
    
    # Branchless Next State Extraction
    sliced = output[:feedback_slice]
    
    if padding_size > 0:
        padding = jnp.zeros((padding_size,))
        next_state = jnp.concatenate([sliced, padding], axis=0)
    else:
        next_state = sliced
            
    return next_state, output


# --- JIT Compiled Wrappers ---

@partial(jax.jit, static_argnames=[
    "n_qubits", "feedback_slice", "padding_size",
    "encoding_strategy", "noise_type"
])
def _step_jit(
    state: jnp.ndarray,
    input_slice: jnp.ndarray,
    reservoir_params: jnp.ndarray,
    measurement_matrix: jnp.ndarray,
    n_qubits: int,
    input_scaling: float,
    feedback_scale: float,
    feedback_slice: int,
    padding_size: int,
    encoding_strategy: str,
    noise_type: str,
    noise_prob: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Standalone JIT wrapper for single step execution.
    """
    return _step_logic(
        state, input_slice, reservoir_params, measurement_matrix,
        n_qubits, input_scaling, feedback_scale,
        feedback_slice, padding_size, encoding_strategy, noise_type, noise_prob
    )

@partial(jax.jit, static_argnames=[
    "n_qubits", "feedback_slice", "padding_size",
    "encoding_strategy", "noise_type"
])
def _forward_jit(
    state_init: jnp.ndarray,
    inputs_time_major: jnp.ndarray,
    reservoir_params: jnp.ndarray,
    measurement_matrix: jnp.ndarray,
    n_qubits: int,
    input_scaling: float,
    feedback_scale: float,
    feedback_slice: int,
    padding_size: int,
    encoding_strategy: str,
    noise_type: str,
    noise_prob: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Forward pass (scan) execution (JIT Compiled).
    Uses uncompiled `_step_logic` inside to ensure proper XLA fusion.
    """
    
    # Partial binding for static/broadcasted args
    step_func = partial(
        _step_logic,
        reservoir_params=reservoir_params,
        measurement_matrix=measurement_matrix,
        n_qubits=n_qubits,
        input_scaling=input_scaling,
        feedback_scale=feedback_scale,
        feedback_slice=feedback_slice,
        padding_size=padding_size,
        encoding_strategy=encoding_strategy,
        noise_type=noise_type,
        noise_prob=noise_prob
    )

    # Vmap over the batch dimension
    step_vmapped = jax.vmap(step_func, in_axes=(0, 0))
    
    # Scan returns (final_carry, stacked_y)
    # y is (next_state, output), so stacked_y is (stacked_states, stacked_outputs) if step returns nested tuple?
    # NO. step returns (next, out). next is new carry. out is y.
    # So stacked_y is stacked_outputs.
    final_carry, stacked_outputs = jax.lax.scan(step_vmapped, state_init, inputs_time_major)
    
    return final_carry, stacked_outputs


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
        input_scaling: float,
        feedback_scale: float,
        measurement_basis: Literal["Z", "ZZ", "Z+ZZ"],
        encoding_strategy: Literal["Rx", "Ry", "Rz", "IQP"] = "Rx",
        noise_type: Literal["clean", "depolarizing", "damping"] = "clean",
        noise_prob: float = 0.0,
    ) -> None:
        """Initialize Quantum Reservoir."""
        # Ensure TC backend and patches are applied (lazy, idempotent)
        _ensure_tensorcircuit_initialized()
        
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
        
        super().__init__(n_units=output_dim, seed=seed)
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_scaling = float(input_scaling)
        self.feedback_scale = float(feedback_scale)
        self.measurement_basis = measurement_basis
        self.n_correlations = n_correlations
        self.encoding_strategy = encoding_strategy
        self.noise_type = noise_type
        self.noise_prob = float(noise_prob)

        self.aggregator = StateAggregator(mode=aggregation_mode)
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
        self.initial_state_vector = jnp.zeros(self.n_qubits)

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
        return jnp.array(matrix_np)

    def initialize_state(self, batch_size: int) -> jnp.ndarray:
        return jnp.zeros((batch_size, self.n_qubits))

    @jaxtyped(typechecker=beartype)
    def step(self, state: Float[Array, "batch n_qubits"], input_data: Float[Array, "batch features"]) -> Tuple[Float[Array, "batch n_qubits"], Float[Array, "batch output_dim"]]:
        """Batched step function for debugging/stepping."""
        # Use vmapped step logic wrapper
        step_func = partial(
            _step_jit, # Use the JIT compiled wrapper
            reservoir_params=self.reservoir_params,
            measurement_matrix=self._measurement_matrix,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            input_scaling=self.input_scaling,
            feedback_scale=self.feedback_scale,
            feedback_slice=self._feedback_slice,
            padding_size=self._padding_size,
            encoding_strategy=self.encoding_strategy,
            noise_type=self.noise_type,
            noise_prob=self.noise_prob
        )
        return jax.vmap(step_func, in_axes=(0, 0))(state, input_data)

    @jaxtyped(typechecker=beartype)
    def forward(self, state: Float[Array, "batch n_qubits"], input_data: Float[Array, "batch time features"]) -> Tuple[Float[Array, "batch n_qubits"], Float[Array, "batch time output_dim"]]:
        """Forward pass using optimized scan."""
        if input_data.ndim != 3:
            raise ValueError(f"Expected (batch, time, feat), got {input_data.shape}")
        
        # Transpose for scan (Time, Batch, Feat)
        inputs_time_major = jnp.swapaxes(input_data, 0, 1)
        
        final_state, stacked_outputs = _forward_jit(
            state,
            inputs_time_major,
            self.reservoir_params,
            self._measurement_matrix,
            self.n_qubits,
            self.input_scaling,
            self.feedback_scale,
            self._feedback_slice,
            self._padding_size,
            self.encoding_strategy,
            self.noise_type,
            self.noise_prob
        )
        
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
        arr = jnp.asarray(inputs, dtype=jnp.float32)
        input_was_2d = (arr.ndim == 2)
        if input_was_2d:
            arr = arr[None, :, :]
        elif arr.ndim != 3:
            raise ValueError(f"QuantumReservoir expects 2D or 3D input, got {arr.shape}")
        
        batch_size = arr.shape[0]
        state = self.initialize_state(batch_size)
        _, states = self.forward(state, arr)
        
        if return_sequences:
            return states[0] if input_was_2d else states
        
        log_label = f"6:{split_name}" if split_name else None
        return self.aggregator.transform(states, log_label=log_label)

    def get_feature_dim(self, time_steps: int) -> int:
        return self.aggregator.get_output_dim(self.n_units, int(time_steps))

    def train(self, inputs: jnp.ndarray, targets: Any = None, **__: Any) -> Dict[str, Any]:
        return {}

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "seed": self.seed,
            "input_scaling": self.input_scaling,
            "feedback_scale": self.feedback_scale,
            "aggregation": self.aggregator.mode.value,
            "measurement_basis": self.measurement_basis,
            "encoding_strategy": self.encoding_strategy,
            "noise_type": self.noise_type,
            "noise_prob": self.noise_prob,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantumReservoir":
        try:
            return cls(
                n_qubits=int(data["n_qubits"]),
                n_layers=int(data["n_layers"]),
                seed=int(data["seed"]),
                input_scaling=float(data.get("input_scaling", 2 * jnp.pi)),
                feedback_scale=float(data.get("feedback_scale", 0.1)),
                aggregation_mode=AggregationMode(data["aggregation"]),
                measurement_basis=data["measurement_basis"],
                encoding_strategy=data.get("encoding_strategy", "Rx"),
                noise_type=data.get("noise_type", "clean"),
                noise_prob=float(data.get("noise_prob", 0.0)),
            )
        except KeyError as exc:
            raise KeyError(f"Missing required quantum reservoir parameter '{exc.args[0]}'") from exc

    def __repr__(self) -> str:
        return (
            f"QuantumReservoir(tc_backend, n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"measurement={self.measurement_basis}, encoding={self.encoding_strategy}, "
            f"noise={self.noise_type}({self.noise_prob}))"
        )


__all__ = ["QuantumReservoir"]
