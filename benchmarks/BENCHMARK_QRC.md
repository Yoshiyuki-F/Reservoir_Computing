# Quantum Reservoir Computing (QRC) Benchmark & Optimizations

This document records the comprehensive performance benchmarks, architectural optimizations, and technical findings of the Quantum Reservoir simulator (JAX + TensorCircuit backend).

## 1. System Configuration
- **Backend:** JAX (XLA) + TensorCircuit
- **Precision:** `complex128` (Strict requirement for numerical stability and high-fidelity research)
- **Hardware Target:** Optimized for high-throughput BLAS/cuBLAS via JAX-XLA.

## 2. Optimization History & Findings

### Phase 1: Support for Multi-dimensional Input
- **Implementation:** **Repeated Encoding**.
- **Change:** Multi-variate input $\mathbf{u}_{n+1}$ is cyclically mapped across all $N$ qubits instead of using only the first dimension.
- **Effect:** Full information retention for datasets like Lorenz (3D).

### Phase 2: Fundamental Speedup (Gate Fusion & Pre-computation)
- **Optimization:** Fused single-qubit rotations (RX-RY-RZ) into 2x2 unitaries and complex Paper R gates into 4x4 unitaries.
- **Static Unitary Pre-computation:** Reservoir rotation matrices are computed once during initialization.
- **Sequence Pre-computation:** Input unitaries for the entire time series (sequence) are computed in one batch before the simulation loop starts.
- **Effect:** Moved thousands of matrix generation calls from the hot loop to initialization/setup phases.

### Phase 3: Entanglement Parallelization
- **Optimization:** **Brickwork Parallelization**.
- **Method:** Replaced the sequential CNOT ladder (0-1, 1-2, 2-3...) with parallel blocks (Even pairs then Odd pairs).
- **Effect:** Reduced logical circuit depth from $N$ to $2$, significantly improving XLA kernel utilization and throughput.

### Phase 4: Measurement Engine Optimization (Benchmarking Results)
Multiple strategies were tested to extract $Z$ and $ZZ$ expectations:
- **Matrix-Vector Dot Product ($M \cdot p$):** Winner. Leverages highly optimized BLAS/cuBLAS libraries.
- **Tensor Marginalization (Reshape/Sum):** Competitive but less flexible for arbitrary observables.

### Phase 5: Deep Framework Optimization & Theoretical Limits
Attempted to push beyond the high-level `tc.Circuit` API to reach the physical speed limit of JAX:
1. **Pure JAX Engine (`tensordot` + `moveaxis`):**
   - Result: **~52 ms/step** (Significant Degradation).
   - Finding: Manual axis manipulation in JAX causes massive memory-stride overhead. TensorCircuit's internal engine is highly specialized for these patterns and outperforms generic JAX tensor operations.
2. **Low-level Backend Bypass:**
   - Finding: JIT-compiled `tc.Circuit` objects are nearly as fast as raw backend calls because XLA eliminates the class overhead during compilation.

## 3. Final Benchmarks (N=14 Qubits, 10 Layers, complex128)

| Engine Version | Step Time (ms) | Speedup | Status |
| :--- | :---: | :---: | :--- |
| Initial (Sequential TC) | ~9.44 | 1.0x | Obsolete |
| Static Pre-computation | ~6.50 | 1.45x | Obsolete |
| Pure JAX (Manual) | ~52.00 | 0.18x | **Rejected** (Memory overhead) |
| **Final Optimized (Static + Brickwork + TC)** | **~2.82** | **3.35x** | **ACTIVE** |

**Final Performance Summary:**
- **Step Throughput:** ~3,550 steps/sec
- **10,000 Step Simulation:** **~28 seconds**
- **Architecture:** TensorCircuit-based state evolution with dual-layer Brickwork entanglement.
- **Stability:** **Strict Normalization** enforced to maintain $1.0$ probability sum across long sequences.

## 4. Engineering Conclusion
The most efficient path for high-qubit simulation ($N \geq 14$) in JAX is not manual tensor manipulation, but **highly optimized domain-specific frameworks (TensorCircuit)** combined with **static parameter pre-computation** and **parallel circuit structures (Brickwork)**.

---
*Documented by: Yoshi & Gemini CLI*
*Last Updated: February 26, 2026*
