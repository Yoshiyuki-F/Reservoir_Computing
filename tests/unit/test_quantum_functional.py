import jax
import jax.numpy as jnp
from reservoir.models.reservoir.quantum.functional import _make_circuit_logic

def test_make_circuit_logic_runs():
    n_qubits = 4
    input_val = jnp.array([0.5, -0.2]) # 2D input
    feedback_val = jnp.array([0.1, 0.2, 0.3, 0.4])
    params = jax.random.uniform(jax.random.key(0), (2, n_qubits, 3)) # 2 layers
    
    probs = _make_circuit_logic(
        input_val=input_val,
        feedback_val=feedback_val,
        params=params,
        n_qubits=n_qubits,
        feedback_scale=1.0,
        noise_type="clean",
        noise_prob=0.0,
        use_remat=False,
        use_reuploading=True,
        rng_key=None
    )
    
    assert probs.shape == (2**n_qubits,)
    print(f"Probabilities sum: {jnp.sum(probs)}")
    assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-5)
    print("Test passed successfully!")

def test_make_circuit_logic_clean_no_reup():
    n_qubits = 2
    input_val = jnp.array([0.5])
    feedback_val = jnp.array([0.1, 0.2])
    params = jax.random.uniform(jax.random.key(0), (1, n_qubits, 3)) # 1 layer
    
    probs = _make_circuit_logic(
        input_val=input_val,
        feedback_val=feedback_val,
        params=params,
        n_qubits=n_qubits,
        feedback_scale=1.0,
        noise_type="clean",
        noise_prob=0.0,
        use_remat=False,
        use_reuploading=False,
        rng_key=None
    )
    
    s = jnp.sum(probs)
    print(f"[Clean No-Reup] Probabilities sum: {s}")
    assert jnp.allclose(s, 1.0, atol=1e-5)

if __name__ == "__main__":
    test_make_circuit_logic_clean_no_reup()
    test_make_circuit_logic_runs()
