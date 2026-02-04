"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/reservoir/quantum/functional.py
Pure functional implementation of Quantum Reservoir logic.
Optimized for JAX JIT compilation.
"""
from __future__ import annotations

from typing import Tuple, Union, Optional, cast
from functools import partial

import jax
import jax.numpy as jnp
import tensorcircuit as tc

from .backend import I_MAT, X_MAT, Y_MAT, Z_MAT


# --- Pure Logic Functions (No JIT Decoration) ---

def _make_circuit_logic(
    inputs: jnp.ndarray,
    params: jnp.ndarray,
    n_qubits: int,

    encoding_strategy: str,
    noise_type: str,
    noise_prob: float,
    use_remat: bool,
    use_reuploading: bool,
    rng_key: Optional[jax.Array] = None
) -> jnp.ndarray:
    """
    Core circuit construction logic with Noise and Encoding support.
    Refactored to use `jax.lax.scan` for the layer loop.
    Compilation time is O(1) with respect to depth.
    """
    is_noisy = (noise_type != "clean")

    def apply_encoding_gate(c_target, idx, val):
        if encoding_strategy == "Rx":
            c_target.rx(idx, theta=val)
        elif encoding_strategy == "Ry":
            c_target.ry(idx, theta=val)
        elif encoding_strategy == "Rz":
            c_target.rz(idx, theta=val)
        elif encoding_strategy == "IQP":
            c_target.h(idx)
            c_target.rz(idx, theta=val)

    # --- 1. Encoding (Input -> State) ---
    c_enc = tc.Circuit(n_qubits)
    # scaled_inputs = inputs * input_scaling # Removed: Projection layer handles scaling
    
    for i in range(n_qubits):
        # jax.debug.print("Qubit {i} Input: {}", inputs[i], i=i)
        apply_encoding_gate(c_enc, i, inputs[i])
            
    # Initial State
    state = c_enc.state()

    # scanに入力する形状を確定させる
    if is_noisy:
        if state.ndim == 1:
            state = jnp.outer(state, jnp.conj(state))
        # ここで確実に (2^N, 2^N) であることを保証


    # --- 2. Dynamics (Layer Scanned) ---
    def layer_step(carry_state, layer_params):
        # Context for MC
        current_key = None
        is_mc = rng_key is not None
        
        if is_mc:
            state_vec, current_key = carry_state
            # In MC mode, we use pure state circuit even if noisy (noise is manual)
            c = tc.Circuit(n_qubits, inputs=state_vec)
        elif is_noisy:
            c = tc.DMCircuit(n_qubits, inputs=carry_state)
        else:
            c = tc.Circuit(n_qubits, inputs=carry_state)

        # Decoherence -  nothing with 1-3p, Px with p, Py with p, Pz with p
        # to simulate NISQ inaccuracy
        def apply_noise_in_layer(indices):
            nonlocal current_key
            if not is_noisy: return
            
            for target_idx in indices:
                if is_mc:
                    # Monte Carlo Noise Injection (Depolarizing)
                    # Use current_key to select Pauli error
                    # Check safe key usage
                    if current_key is None:
                         # Should not happen in MC mode, but safe guard
                         continue
                         
                    k1, current_key = jax.random.split(current_key)
                    r = jax.random.uniform(k1)
                    
                    # Depolarizing Channel: (1-p)I, p/3 X, p/3 Y, p/3 Z
                    p = noise_prob
                    
                    # Branchless selection of matrix index
                    # 0:I, 1:X, 2:Y, 3:Z
                    gate_idx = jnp.zeros((), dtype=jnp.int32)
                    gate_idx = jax.lax.select(r < 3*p, jnp.array(3, dtype=jnp.int32), gate_idx) # Z
                    gate_idx = jax.lax.select(r < 2*p, jnp.array(2, dtype=jnp.int32), gate_idx) # Y
                    gate_idx = jax.lax.select(r < p, jnp.array(1, dtype=jnp.int32), gate_idx)   # X
                    
                    # Switch mechanism to get matrix
                    def get_mat_i(_): return I_MAT
                    def get_mat_x(_): return X_MAT
                    def get_mat_y(_): return Y_MAT
                    def get_mat_z(_): return Z_MAT
                    
                    mat = jax.lax.switch(gate_idx, [get_mat_i, get_mat_x, get_mat_y, get_mat_z], None)
                    c.unitary(target_idx, unitary=mat, name="mc_noise")
                else:
                    # Density Matrix Noise
                    if noise_type == "depolarizing":
                        c.depolarizing(target_idx, px=noise_prob, py=noise_prob, pz=noise_prob)
                    elif noise_type == "damping":
                        c.amplitude_damping(target_idx, gamma=noise_prob)

        # === [A] Re-uploading (Optinoal) ===
        if use_reuploading:
            # Re-apply encoding gates with scaled inputs
            # Note: inputs are captured from closure
            for idx in range(n_qubits):
                apply_encoding_gate(c, idx, inputs[idx])
                apply_noise_in_layer([idx])

        # === [B] CNOT Ladder ===
        for j in range(n_qubits - 1):
            c.cnot(j, j + 1)
            apply_noise_in_layer([j, j + 1])
            
        # === [C] Random Rotations ===
        for k in range(n_qubits):
            c.rx(k, theta=layer_params[k, 0])
            apply_noise_in_layer([k])
            
            c.ry(k, theta=layer_params[k, 1])
            apply_noise_in_layer([k])
            
            c.rz(k, theta=layer_params[k, 2])
            apply_noise_in_layer([k])
            
        # === [D] Reverse Ladder ===
        for l in range(n_qubits - 2, -1, -1):
            c.cnot(l, l + 1)
            apply_noise_in_layer([l, l + 1])
            
        new_state = c.state()
        
        if is_mc:
            return (new_state, current_key), None
        return new_state, None

    if use_remat:
        layer_step = jax.checkpoint(layer_step)

    # Scan over layers
    # Initial carry setup
    if rng_key is not None:
        init_carry = (state, rng_key)
    else:
        init_carry = state
        
    final_carry, _ = jax.lax.scan(layer_step, init_carry, params)
    
    # Unpack Logic
    if rng_key is not None:
        final_state, final_key = final_carry
    else:
        final_state = final_carry
    
    # --- 3. Measurement ---
    if is_noisy and not is_mc:
        # final_state is Density Matrix (2^N, 2^N)
        # Probabilities are diagonal elements.
        probs = jnp.real(jnp.diag(cast(jnp.ndarray, final_state)))
        return probs.astype(jnp.float32)
    else:
        # Pure State (Clean or MC)
        # final_state is Vector (2^N,)
        return (jnp.abs(cast(jnp.ndarray, final_state)) ** 2).astype(jnp.float32)


def _step_logic(
    state: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    input_slice: jnp.ndarray,
    reservoir_params: jnp.ndarray,
    measurement_matrix: jnp.ndarray,
    n_qubits: int,

    leak_rate: float,
    feedback_slice: int,
    padding_size: int,
    encoding_strategy: str,
    noise_type: str,
    noise_prob: float,
    use_remat: bool,
    use_reuploading: bool
) -> Tuple[Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], jnp.ndarray]:
    """
    Core step logic (Single step).
    NOT JIT compiled here. Used by both standalone JIT and Scan JIT functions.
    Handles both clean state (Array) and MC state (Tuple[Array, Key]).
    """
    # Polymorphic State Unpacking
    rng_key = None
    if isinstance(state, (tuple, list)):
        state_vec, rng_key = state
    else:
        state_vec = state

    # Prepare RNG for this step and next step
    step_key = None
    next_key = None
    if rng_key is not None:
        step_key, next_key = jax.random.split(rng_key)

    # Circuit Execution
    # Li-ESN: Input is just u_t, no feedback injection here
    probs = _make_circuit_logic(
        input_slice, 
        reservoir_params, 
        n_qubits, 
        encoding_strategy,
        noise_type,
        noise_prob,
        use_remat,
        use_reuploading,
        step_key
    )
    
    # Vectorized Measurement: E = M @ p
    output = jnp.dot(measurement_matrix, probs)
    
    # Branchless Next State Extraction
    measured_state = output[:feedback_slice]
    
    if padding_size > 0:
        padding = jnp.zeros((padding_size,), dtype=jnp.float32)
        measured_state = jnp.concatenate([measured_state, padding], axis=0)

    # Li-ESN State Update:
    # x_t = (1 - alpha) * x_{t-1} + alpha * Measure(Circuit(u_t))
    # next_state_vec = (1.0 - leak_rate) * state_vec + leak_rate * measured_state
    next_state_vec = (1.0 - leak_rate) * state_vec + 1.2 * measured_state


    if next_key is not None:
        return (next_state_vec, cast(jnp.ndarray, next_key)), output
        
    return next_state_vec, output


# --- JIT Compiled Wrappers ---

@partial(jax.jit, static_argnames=[
    "n_qubits", "feedback_slice", "padding_size",
    "encoding_strategy", "noise_type", "use_remat", "use_reuploading"
])
def _step_jit(
    state: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    input_slice: jnp.ndarray,
    reservoir_params: jnp.ndarray,
    measurement_matrix: jnp.ndarray,
    n_qubits: int,
    leak_rate: float,
    feedback_slice: int,
    padding_size: int,
    encoding_strategy: str,
    noise_type: str,
    noise_prob: float,
    use_remat: bool,
    use_reuploading: bool
) -> Tuple[Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], jnp.ndarray]:
    """
    Standalone JIT wrapper for single step execution.
    """
    return _step_logic(
        state, input_slice, reservoir_params, measurement_matrix,
        n_qubits, leak_rate,
        feedback_slice, padding_size, encoding_strategy, noise_type, noise_prob, use_remat, use_reuploading
    )

@partial(jax.jit, static_argnames=[
    "n_qubits", "feedback_slice", "padding_size",
    "encoding_strategy", "noise_type", "use_remat", "use_reuploading"
])
def _forward_jit(
    state_init: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    inputs_time_major: jnp.ndarray,
    reservoir_params: jnp.ndarray,
    measurement_matrix: jnp.ndarray,
    n_qubits: int,

    leak_rate: float,
    feedback_slice: int,
    padding_size: int,
    encoding_strategy: str,
    noise_type: str,
    noise_prob: float,
    use_remat: bool,
    use_reuploading: bool
) -> Tuple[Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], jnp.ndarray]:
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
        leak_rate=leak_rate,
        feedback_slice=feedback_slice,
        padding_size=padding_size,
        encoding_strategy=encoding_strategy,
        noise_type=noise_type,
        noise_prob=noise_prob,
        use_remat=use_remat,
        use_reuploading=use_reuploading
    )

    # Vmap over the batch dimension
    step_vmapped = jax.vmap(step_func, in_axes=(0, 0))
    
    # Scan returns (final_carry, stacked_y)
    # y is (next_state, output), so stacked_y is (stacked_states, stacked_outputs) if step returns nested tuple?
    # NO. step returns (next, out). next is new carry. out is y.
    # So stacked_y is stacked_outputs.
    final_carry, stacked_outputs = jax.lax.scan(step_vmapped, state_init, inputs_time_major)
    
    return final_carry, stacked_outputs
