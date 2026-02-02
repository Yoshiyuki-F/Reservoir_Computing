"""
Backend configuration and constants for Quantum Reservoir Computing.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import tensorcircuit as tc

# --- Lazy Initialization for Safety & Isolation ---
_TC_INITIALIZED = False

def _ensure_tensorcircuit_initialized(precision: str = "complex64") -> None:
    """
    Lazily configure TensorCircuit and patch Numpy.
    Ensures global side effects only happen when QuantumReservoir is verified to be used.
    """
    global _TC_INITIALIZED
    
    # Enable x64 if needed for complex128
    if precision == "complex128":
        jax.config.update("jax_enable_x64", True)

    if _TC_INITIALIZED:
        if precision != tc.dtypestr:
            tc.set_dtype(precision)
        return

    # Configure TensorCircuit
    # Localize backend setting to avoid affecting other modules on import
    tc.set_backend("jax")
    tc.set_dtype(precision)

    _TC_INITIALIZED = True

# Pauli Constants for Monte Carlo Simulation (Manual Noise)
I_MAT = jnp.eye(2, dtype=jnp.complex64)
X_MAT = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
Y_MAT = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
Z_MAT = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
