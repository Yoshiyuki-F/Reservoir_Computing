import jax
import jax.numpy as jnp
import time

jax.config.update("jax_enable_x64", True)

theta = jax.random.uniform(jax.random.key(0), (10000,))

@jax.jit
def old_way(theta):
    c = jnp.cos(theta / 2.0)
    s = -1.0j * jnp.sin(theta / 2.0)
    zm = jnp.exp(-1.0j * theta / 2.0)
    zp = jnp.exp(1.0j * theta / 2.0)
    return c, s, zm, zp

@jax.jit
def new_way(theta):
    zp = jnp.exp(1.0j * (theta / 2.0))
    zm = jnp.conj(zp)
    c = jnp.real(zp)
    s = -1.0j * jnp.imag(zp)
    return c, s, zm, zp

old_way(theta)
new_way(theta)

t0 = time.time()
for _ in range(10000):
    c1, s1, zm1, zp1 = old_way(theta)
jax.block_until_ready(c1)
print("Old way:", time.time() - t0)

t0 = time.time()
for _ in range(10000):
    c2, s2, zm2, zp2 = new_way(theta)
jax.block_until_ready(c2)
print("New way:", time.time() - t0)

print("Max diff c:", jnp.max(jnp.abs(c1 - c2)))
print("Max diff s:", jnp.max(jnp.abs(s1 - s2)))
print("Max diff zm:", jnp.max(jnp.abs(zm1 - zm2)))
print("Max diff zp:", jnp.max(jnp.abs(zp1 - zp2)))
