import jax
import jax.numpy as jnp
import tensorcircuit as tc
import time

tc.set_backend('jax')
jax.config.update("jax_enable_x64", True)

n_qubits = 14
# dummy unitaries
key = jax.random.key(0)
u1 = jax.random.normal(key, (n_qubits, 2, 2)) + 1j * jax.random.normal(key, (n_qubits, 2, 2))

# Make them unitary
for i in range(n_qubits):
    q, r = jnp.linalg.qr(u1[i])
    u1 = u1.at[i].set(q)

@jax.jit
def no_fusion(state):
    c = tc.Circuit(n_qubits, inputs=state)
    for j in range(1, n_qubits - 1, 2):
        c.cnot(j, j + 1)
    for k in range(n_qubits):
        c.unitary(k, unitary=u1[k])
    for j in range(1, n_qubits - 1, 2):
        c.cnot(j, j + 1)
    return c.state()

# Precompute fused unitaries for odd pairs
# CNOT matrix
cx = jnp.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=jnp.complex128)
fused_odd = []
for j in range(1, n_qubits - 1, 2):
    r1 = u1[j]
    r2 = u1[j+1]
    # R1 \otimes R2
    kron = jnp.kron(r1, r2)
    # CX @ (R1 \otimes R2) @ CX
    fused = cx @ kron @ cx
    fused_odd.append(fused)
fused_odd = jnp.stack(fused_odd)

@jax.jit
def with_fusion(state):
    c = tc.Circuit(n_qubits, inputs=state)
    # applies R for even qubits that are not in odd pairs (qubit 0 and possibly N-1)
    c.unitary(0, unitary=u1[0])
    if n_qubits % 2 == 0:
        c.unitary(n_qubits - 1, unitary=u1[n_qubits - 1])
        
    idx = 0
    for j in range(1, n_qubits - 1, 2):
        c.unitary(j, j+1, unitary=fused_odd[idx])
        idx += 1
    return c.state()

s0 = jnp.zeros(2**n_qubits, dtype=jnp.complex128)
s0 = s0.at[0].set(1.0)

# Check correctness
s_no = no_fusion(s0)
s_yes = with_fusion(s0)
print("Max diff:", jnp.max(jnp.abs(s_no - s_yes)))

# Benchmark
no_fusion(s0).block_until_ready()
t0 = time.time()
for _ in range(1000):
    no_fusion(s0)
jax.block_until_ready(s_no)
print("No fusion:", time.time() - t0)

with_fusion(s0).block_until_ready()
t0 = time.time()
for _ in range(1000):
    with_fusion(s0)
jax.block_until_ready(s_yes)
print("With fusion:", time.time() - t0)

