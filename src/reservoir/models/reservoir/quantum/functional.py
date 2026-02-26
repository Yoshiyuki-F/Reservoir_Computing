"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/reservoir/quantum/functional.py
Pure functional implementation of Quantum Reservoir logic.
Final High-Performance Engine: Static Pre-computation + Brickwork Parallelization.
Optimized for complex128 precision.
"""
from __future__ import annotations

from typing import cast, TYPE_CHECKING
from functools import partial
import os

import jax
import jax.numpy as jnp
import tensorcircuit as tc
import time

from .backend import I_MAT, X_MAT, Y_MAT, Z_MAT

if TYPE_CHECKING:
    from reservoir.core.types import JaxF64, JaxKey


# --- Performance Optimization Helpers ---

def _get_paper_R_unitary(theta: JaxF64) -> JaxF64:
    """Compute fused 4x4 unitary for the Paper R gate (vmap safe)."""
    c = jnp.cos(theta / 2.0); isin = 1.0j * jnp.sin(theta / 2.0)
    e_neg = jnp.exp(-1.0j * theta / 2.0); e_pos = jnp.exp(1.0j * theta / 2.0)
    cx = jnp.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=jnp.complex128)
    
    rx = jnp.stack([jnp.stack([c, -isin], axis=-1), jnp.stack([-isin, c], axis=-1)], axis=-2)
    rz = jnp.stack([
        jnp.stack([e_neg, jnp.zeros_like(theta, dtype=jnp.complex128)], axis=-1),
        jnp.stack([jnp.zeros_like(theta, dtype=jnp.complex128), e_pos], axis=-1)
    ], axis=-2)
    
    rx_2q = jnp.kron(rx, rx); rz_2q = jnp.kron(jnp.eye(2, dtype=jnp.complex128), rz)
    m = jnp.matmul(cx, rx_2q); m = jnp.matmul(rz_2q, m); m = jnp.matmul(cx, m)
    
    from .backend import _ensure_tensorcircuit_initialized
    _ensure_tensorcircuit_initialized()
    return tc.backend.convert_to_tensor(m)


def _get_fused_rotation_matrix(params: JaxF64) -> JaxF64:
    """Fuse RX, RY, RZ rotations into a single 2x2 unitary matrix."""
    tx, ty, tz = params[0], params[1], params[2]
    def rot_x(p): c = jnp.cos(p/2); s = -1.0j*jnp.sin(p/2); return jnp.stack([jnp.stack([c, s], axis=-1), jnp.stack([s, c], axis=-1)], axis=-2)
    def rot_y(p): c = jnp.cos(p/2); s = jnp.sin(p/2); return jnp.stack([jnp.stack([c, -s], axis=-1), jnp.stack([s, c], axis=-1)], axis=-2)
    def rot_z(p): 
        en = jnp.exp(-1.0j*p/2); ep = jnp.exp(1.0j*p/2); z = jnp.zeros_like(p, dtype=jnp.complex128)
        return jnp.stack([jnp.stack([en, z], axis=-1), jnp.stack([z, ep], axis=-1)], axis=-2)
    m = jnp.matmul(rot_z(tz), jnp.matmul(rot_y(ty), rot_x(tx)))
    from .backend import _ensure_tensorcircuit_initialized
    _ensure_tensorcircuit_initialized()
    return tc.backend.convert_to_tensor(m)


# --- Core Step Logic ---

def _make_circuit_logic(
    input_unitaries: JaxF64, # (N, 4, 4)
    feedback_val: JaxF64,
    params: JaxF64,          # (L, N, 2, 2)
    n_qubits: int,
    feedback_scale: float,
    noise_type: str,
    noise_prob: float,
    use_remat: bool,
    use_reuploading: bool,
    rng_key: JaxKey | None = None
) -> JaxF64:
    tc.set_backend("jax")
    is_noisy = (noise_type != "clean")
    is_mc = rng_key is not None

    fb_unitaries = jax.vmap(_get_paper_R_unitary)(feedback_val * feedback_scale)

    c = tc.Circuit(n_qubits)
    # Encoding: Input then Feedback (sequential applied to same pairs)
    for i in range(n_qubits):
        c.unitary(i, (i + 1) % n_qubits, unitary=input_unitaries[i])
        c.unitary((i + 1) % n_qubits, i, unitary=fb_unitaries[i])

    state = c.state()
    if is_noisy and state.ndim == 1: state = jnp.outer(state, jnp.conj(state))

    def layer_step(carry_state, layer_rotation_unitaries):
        current_key = None
        if is_mc: state_vec, current_key = carry_state; lc = tc.Circuit(n_qubits, inputs=state_vec)
        elif is_noisy: lc = tc.DMCircuit(n_qubits, inputs=carry_state)
        else: lc = tc.Circuit(n_qubits, inputs=carry_state)

        def apply_noise(indices):
            nonlocal current_key
            if not is_noisy: return
            for idx in indices:
                if is_mc:
                    if current_key is None: continue
                    k1, current_key = jax.random.split(current_key); r = jax.random.uniform(k1)
                    gate_idx = jnp.zeros((), dtype=jnp.int32)
                    gate_idx = jax.lax.select(r < 3*noise_prob, jnp.array(3), gate_idx)
                    gate_idx = jax.lax.select(r < 2*noise_prob, jnp.array(2), gate_idx)
                    gate_idx = jax.lax.select(r < noise_prob, jnp.array(1), gate_idx)
                    mat = jax.lax.switch(gate_idx, [lambda _: I_MAT, lambda _: X_MAT, lambda _: Y_MAT, lambda _: Z_MAT], None)
                    lc.unitary(idx, unitary=mat)
                else:
                    if noise_type == "depolarizing": lc.depolarizing(idx, px=noise_prob, py=noise_prob, pz=noise_prob)
                    elif noise_type == "damping": lc.amplitude_damping(idx, gamma=noise_prob)

        if use_reuploading:
            for i in range(n_qubits): lc.unitary(i, (i + 1) % n_qubits, unitary=input_unitaries[i])
            for idx in range(n_qubits): apply_noise([idx])

        # Brickwork Entanglement
        for j in range(0, n_qubits - 1, 2): lc.cnot(j, j + 1); apply_noise([j, j + 1])
        for j in range(1, n_qubits - 1, 2): lc.cnot(j, j + 1); apply_noise([j, j + 1])
        for k in range(n_qubits): lc.unitary(k, unitary=layer_rotation_unitaries[k]); apply_noise([k])
        for j in range(1, n_qubits - 1, 2): lc.cnot(j, j + 1); apply_noise([j, j + 1])
        for j in range(0, n_qubits - 1, 2): lc.cnot(j, j + 1); apply_noise([j, j + 1])
            
        new_state = lc.state()
        return (new_state, current_key) if is_mc else new_state, None

    if use_remat: layer_step = jax.checkpoint(layer_step)
    final_carry, _ = jax.lax.scan(layer_step, (state, rng_key) if is_mc else state, params)
    final_state = final_carry[0] if is_mc else final_carry
    
    if is_noisy and not is_mc: probs = jnp.real(jnp.diag(cast("JaxF64", final_state)))
    else: probs = jnp.abs(cast("JaxF64", final_state)) ** 2
    return probs / (jnp.sum(probs) + 1e-12)


def _step_logic(state, step_input_unitaries, reservoir_params, measurement_matrix, n_qubits, feedback_scale, noise_type, noise_prob, use_remat, use_reuploading):
    state_vec, rng_key = state
    step_key = None; next_key = None
    if rng_key is not None: step_key, next_key = jax.random.split(rng_key)
    probs = _make_circuit_logic(step_input_unitaries, state_vec, reservoir_params, n_qubits, feedback_scale, noise_type, noise_prob, use_remat, use_reuploading, step_key)
    output = jnp.dot(measurement_matrix, probs)
    return (output[:n_qubits], next_key), output


# --- Public API Wrappers ---

@partial(jax.jit, static_argnames=["n_qubits", "noise_type", "use_remat", "use_reuploading"])
def _step_jit(state, input_slice, reservoir_params, measurement_matrix, n_qubits, feedback_scale, noise_type, noise_prob, use_remat, use_reuploading):
    input_unitaries = jax.vmap(_get_paper_R_unitary)(input_slice[jnp.arange(n_qubits) % input_slice.shape[0]])
    return _step_logic(state, input_unitaries, reservoir_params, measurement_matrix, n_qubits, feedback_scale, noise_type, noise_prob, use_remat, use_reuploading)

@partial(jax.jit, static_argnames=["n_qubits", "noise_type", "use_remat", "use_reuploading"])
def _chunk_scan_jit(state_carry, chunk_input_unitaries, reservoir_params, measurement_matrix, n_qubits, feedback_scale, noise_type, noise_prob, use_remat, use_reuploading):
    step_func = partial(_step_logic, reservoir_params=reservoir_params, measurement_matrix=measurement_matrix, n_qubits=n_qubits, feedback_scale=feedback_scale, noise_type=noise_type, noise_prob=noise_prob, use_remat=use_remat, use_reuploading=use_reuploading)
    
    in_axs = (0, 0) if chunk_input_unitaries.ndim == 5 else (0, None)
    return jax.lax.scan(jax.vmap(step_func, in_axes=in_axs), state_carry, chunk_input_unitaries)

def _forward_jit(state_init, inputs_time_major, reservoir_params, measurement_matrix, n_qubits, feedback_scale, noise_type, noise_prob, use_remat, use_reuploading, chunk_size):
    T, B, D = inputs_time_major.shape
    def get_step_unitaries(u_vec): return jax.vmap(_get_paper_R_unitary)(u_vec[jnp.arange(n_qubits) % D])
    
    start_time = time.time()
    # 1. Pre-compute all unitaries for the entire sequence (Time, Batch, N, 4, 4)
    all_input_unitaries = jax.vmap(jax.vmap(get_step_unitaries))(inputs_time_major)
    
    # 2. Execute full sequence in one shot (No chunking boilerplate)
    print(f"[functional.py] Starting _forward_jit execution (T={T}, Batch={B}) at {time.strftime('%H:%M:%S', time.localtime(start_time))}...")
    final_carry, stacked_outputs = _chunk_scan_jit(
        state_init, all_input_unitaries, 
        reservoir_params, measurement_matrix, 
        n_qubits, feedback_scale, use_reuploading
    )
    print(f"[functional.py] _forward_jit execution finished in {time.time() - start_time:.4f} seconds.")
    
    return final_carry, stacked_outputs.reshape(T, B, -1)
