"""
Benchmark: Dense vs Sparse Measurement Matrix
Verifies if jax.experimental.sparse offers benefits for the Measurement Matrix.
"""
import time
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import numpy as np
import pandas as pd

def get_memory_usage(obj):
    """Estimate memory usage of a JAX array."""
    if hasattr(obj, "nbytes"):
        return obj.nbytes
    elif hasattr(obj, "data") and hasattr(obj, "indices"): # BCOO
        return obj.data.nbytes + obj.indices.nbytes
    return 0

def compute_measurement_matrix_dense(n_qubits, measurement_basis="Z+ZZ"):
    """Original vectorized implementation."""
    dim = 2 ** n_qubits
    basis_states = jnp.arange(dim)
    
    shifts = n_qubits - 1 - jnp.arange(n_qubits)
    bits = (basis_states[:, None] >> shifts[None, :]) & 1
    z_values = (1 - 2 * bits)
    
    row_blocks = []
    if measurement_basis in ("Z", "Z+ZZ"):
        row_blocks.append(z_values.T)
    if measurement_basis in ("ZZ", "Z+ZZ"):
        idx_i, idx_j = jnp.triu_indices(n_qubits, k=1)
        zz_values = z_values[:, idx_i] * z_values[:, idx_j]
        row_blocks.append(zz_values.T)
        
    return jnp.vstack(row_blocks)

def compute_measurement_matrix_sparse(n_qubits, measurement_basis="Z+ZZ"):
    """
    Constructs BCOO matrix. 
    Since the matrix is fully dense (+1/-1), we expect sparsity to be 0%.
    This serves as a verification baseline.
    """
    dense = compute_measurement_matrix_dense(n_qubits, measurement_basis)
    return sparse.BCOO.fromdense(dense)

@jax.jit
def dot_dense(mat, vec):
    return jnp.dot(mat, vec)

@jax.jit
def dot_sparse(mat, vec):
    # sparse.dot or mat @ vec
    # sparse matrix multiplication
    return mat @ vec

def run_benchmark():
    results = []
    qubits_range = [10, 12, 14, 16] # 18 might take too long for interactive check
    
    print(f"{'N':<5} | {'Type':<8} | {'Mem (MB)':<10} | {'Build (s)':<10} | {'Dot (ms)':<10} | {'Correct'}")
    print("-" * 70)
    
    for n in qubits_range:
        dim = 2**n
        prob_vec = jax.random.normal(jax.random.key(0), (dim,))
        prob_vec = jnp.abs(prob_vec) / jnp.sum(jnp.abs(prob_vec))
        
        # --- Dense ---
        start = time.time()
        mat_dense = compute_measurement_matrix_dense(n)
        jax.block_until_ready(mat_dense)
        build_time_dense = time.time() - start
        
        mem_dense = get_memory_usage(mat_dense) / 1024**2
        
        # Warmup
        _ = dot_dense(mat_dense, prob_vec)
        
        start = time.time()
        res_dense = dot_dense(mat_dense, prob_vec)
        jax.block_until_ready(res_dense)
        dot_time_dense = (time.time() - start) * 1000
        
        results.append({
            "N": n, "Type": "Dense", "Mem_MB": mem_dense, 
            "Build_s": build_time_dense, "Dot_ms": dot_time_dense, "Correct": True
        })
        print(f"{n:<5} | {'Dense':<8} | {mem_dense:<10.2f} | {build_time_dense:<10.4f} | {dot_time_dense:<10.4f} | {True}")

        # --- Sparse ---
        try:
            start = time.time()
            mat_sparse = compute_measurement_matrix_sparse(n)
            jax.block_until_ready(mat_sparse.data)
            build_time_sparse = time.time() - start
            
            mem_sparse = get_memory_usage(mat_sparse) / 1024**2
            
            # Warmup
            _ = dot_sparse(mat_sparse, prob_vec)
            
            start = time.time()
            res_sparse = dot_sparse(mat_sparse, prob_vec)
            jax.block_until_ready(res_sparse)
            dot_time_sparse = (time.time() - start) * 1000
            
            # Check correctness
            correct = np.allclose(res_dense, res_sparse, atol=1e-5)
            
            results.append({
                "N": n, "Type": "Sparse", "Mem_MB": mem_sparse, 
                "Build_s": build_time_sparse, "Dot_ms": dot_time_sparse, "Correct": correct
            })
            print(f"{n:<5} | {'Sparse':<8} | {mem_sparse:<10.2f} | {build_time_sparse:<10.4f} | {dot_time_sparse:<10.4f} | {correct}")
            
        except Exception as e:
            print(f"{n:<5} | {'Sparse':<8} | {'ERROR':<10} | {'-':<10} | {'-':<10} | {str(e)}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_benchmark()
    df.to_csv("benchmarks/sparse_results.csv", index=False)
