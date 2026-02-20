"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/reservoir/quantum/functional.py
Pure functional implementation of Quantum Reservoir logic.
Optimized for JAX JIT compilation.

Feedback QRC (Murauer et al., 2025):
  - Input and feedback are injected via separate R gates (no additive mixing)
  - State is reset each step; memory is carried only via measurement feedback
  - R_{i,j}(θ) = CX_{ij} RZ_j(θ) CX_{ij} RX_j(θ) RX_i(θ)
"""
from __future__ import annotations

from typing import Tuple, Union, Optional, cast
from functools import partial

import jax
import jax.numpy as jnp
from reservoir.core.types import JaxF64
import tensorcircuit as tc

from .backend import I_MAT, X_MAT, Y_MAT, Z_MAT


# --- Paper R Gate (Murauer et al., 2025 Eq.1) ---

def _apply_paper_R_gate(c, i: int, j: int, val, scaling: float):
    """
    論文 Eq.(1) の R_{i,j}(theta) ゲート.
    R_{i,j}(θ) = CX_{ij} · RZ_j(θ) · CX_{ij} · RX_j(θ) · RX_i(θ)
    適用順序は右から左: RX_i → RX_j → CX → RZ_j → CX
    """
    theta = val * scaling
    c.rx(i, theta=theta)
    c.rx(j, theta=theta)
    c.cnot(i, j)
    c.rz(j, theta=theta)
    c.cnot(i, j)


# --- Pure Logic Functions (No JIT Decoration) ---

def _make_circuit_logic(
    input_val: JaxF64,       # 現在の入力 u_t (size: n_qubits)
    feedback_val: JaxF64,    # 前回の測定値 m_{t-1} (size: n_qubits)
    params: JaxF64,
    n_qubits: int,
    feedback_scale: float,        # a_fb: R gate feedback scaling

    noise_type: str,
    noise_prob: float,
    use_remat: bool,
    use_reuploading: bool,
    rng_key: Optional[jax.Array] = None
) -> JaxF64:
    """
    Core circuit construction logic with Feedback QRC architecture.

    Architecture (Murauer et al., 2025):
      1. Input Projection: R gate with s_k (pre-scaled by MinMaxScaler)
      2. Feedback Projection: N R gates with a_fb * m_{k-1}^j
      3. Reservoir Dynamics: HEA layers (CNOT ladder + random rotations)
      4. Measurement: probability vector

    Compilation time is O(1) with respect to depth via jax.lax.scan.
    """
    is_noisy = (noise_type != "clean")

    # --- 1. Input + Feedback Projection (R gates) ---
    c_enc = tc.Circuit(n_qubits)

    # Input Projection: R gate with pre-scaled value (a_in applied by MinMaxScaler)
    if n_qubits >= 2:
        _apply_paper_R_gate(c_enc, 0, 1, input_val[0], 1.0)
    else:
        c_enc.rx(0, theta=input_val[0])

    # Feedback Projection: Apply N R gates, one per qubit
    # Each qubit's previous measurement result is injected via its own R gate
    if n_qubits >= 2:
        for i in range(n_qubits):
            target = i
            control = (i + 1) % n_qubits
            _apply_paper_R_gate(c_enc, control, target, feedback_val[i], feedback_scale)
    else:
        c_enc.rx(0, theta=feedback_val[0] * feedback_scale)

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
                    if current_key is None:
                         continue
                         
                    k1, current_key = jax.random.split(current_key)
                    r = jax.random.uniform(k1)
                    
                    p = noise_prob
                    
                    gate_idx = jnp.zeros((), dtype=jnp.int32)
                    gate_idx = jax.lax.select(r < 3*p, jnp.array(3, dtype=jnp.int32), gate_idx) # Z
                    gate_idx = jax.lax.select(r < 2*p, jnp.array(2, dtype=jnp.int32), gate_idx) # Y
                    gate_idx = jax.lax.select(r < p, jnp.array(1, dtype=jnp.int32), gate_idx)   # X
                    
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

        # === [A] Re-uploading (Optional) ===
        if use_reuploading:
            # Re-inject pre-scaled input via R gate (a_in already applied by MinMaxScaler)
            if n_qubits >= 2:
                _apply_paper_R_gate(c, 0, 1, input_val[0], 1.0)
            else:
                c.rx(0, theta=input_val[0])
            for idx in range(n_qubits):
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
        probs = jnp.real(jnp.diag(cast(JaxF64, final_state)))
        return probs.astype(jnp.float_)
    else:
        # Pure State (Clean or MC)
        # final_state is Vector (2^N,)
        return (jnp.abs(cast(JaxF64, final_state)) ** 2).astype(jnp.float_)


def _step_logic(
    state: Union[JaxF64, Tuple[JaxF64, JaxF64]],
    input_slice: JaxF64,
    reservoir_params: JaxF64,
    measurement_matrix: JaxF64,
    n_qubits: int,
    feedback_scale: float,
    noise_type: str,
    noise_prob: float,
    use_remat: bool,
    use_reuploading: bool
) -> Tuple[Union[JaxF64, Tuple[JaxF64, JaxF64]], JaxF64]:
    """
    Core step logic - Feedback QRC (Murauer et al., 2025).
    
    Feedback QRC Equation:
      1. Circuit Phase (Input + Feedback via separate R gates):
         p_t = |Circuit(R_in(s_k), R_fb(m_{k-1}), U_res)|²
         
      2. Measurement Phase:
         z_t = M @ p_t   (expectation values)
         
      3. State Update (Resetting):
         m_t = z_t[:n_qubits]  (measurement → next feedback)
         No leak rate blending. Memory is carried solely through R gate feedback.
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

    # === Phase 1+2: Circuit Execution (Input & Feedback via separate R gates) ===
    probs = _make_circuit_logic(
        input_val=input_slice,
        feedback_val=state_vec,      # 前回の測定値がフィードバック
        params=reservoir_params,
        n_qubits=n_qubits,
        feedback_scale=feedback_scale,
        noise_type=noise_type,
        noise_prob=noise_prob,
        use_remat=use_remat,
        use_reuploading=use_reuploading,
        rng_key=step_key
    )
    
    # Vectorized Measurement: z_t = M @ p
    output = jnp.dot(measurement_matrix, probs)
    
    # === Phase 3: State Update (Resetting) ===
    # No leak rate blending. Measurement result directly becomes next feedback.
    next_state_vec = output[:n_qubits]

    if next_key is not None:
        return (next_state_vec, cast(JaxF64, next_key)), output
        
    return next_state_vec, output


# --- JIT Compiled Wrappers ---

@partial(jax.jit, static_argnames=[
    "n_qubits", "noise_type", "use_remat", "use_reuploading"
])
def _step_jit(
    state: Union[JaxF64, Tuple[JaxF64, JaxF64]],
    input_slice: JaxF64,
    reservoir_params: JaxF64,
    measurement_matrix: JaxF64,
    n_qubits: int,
    feedback_scale: float,
    noise_type: str,
    noise_prob: float,
    use_remat: bool,
    use_reuploading: bool
) -> Tuple[Union[JaxF64, Tuple[JaxF64, JaxF64]], JaxF64]:
    """
    Standalone JIT wrapper for single step execution.
    """
    return _step_logic(
        state, input_slice, reservoir_params, measurement_matrix,
        n_qubits, feedback_scale,
        noise_type, noise_prob, use_remat, use_reuploading
    )

@partial(jax.jit, static_argnames=[
    "n_qubits", "noise_type", "use_remat", "use_reuploading"
])
def _forward_jit(
    state_init: Union[JaxF64, Tuple[JaxF64, JaxF64]],
    inputs_time_major: JaxF64,
    reservoir_params: JaxF64,
    measurement_matrix: JaxF64,
    n_qubits: int,
    feedback_scale: float,
    noise_type: str,
    noise_prob: float,
    use_remat: bool,
    use_reuploading: bool
) -> Tuple[Union[JaxF64, Tuple[JaxF64, JaxF64]], JaxF64]:
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
        feedback_scale=feedback_scale,
        noise_type=noise_type,
        noise_prob=noise_prob,
        use_remat=use_remat,
        use_reuploading=use_reuploading
    )

    # Vmap over the batch dimension
    step_vmapped = jax.vmap(step_func, in_axes=(0, 0))
    
    # Scan returns (final_carry, stacked_y)
    final_carry, stacked_outputs = jax.lax.scan(step_vmapped, state_init, inputs_time_major)
    
    return final_carry, stacked_outputs
