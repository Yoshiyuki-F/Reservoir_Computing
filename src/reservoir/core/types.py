"""
reservoir/core/types.py — Central Type Definitions & Domain Gateway

厳格な型エイリアス定義。AnyやUnionは一切禁止。
各ドメインのファイルはここからimportし、迷わず正しい型を使う。

NUMPY Domain → NpF64 (import numpy のみ使うファイル用)
JAX Domain   → JaxF64 (import jax のみ使うファイル用)

CPU↔GPU転送 → to_jax_f64() / to_np_f64() を必ず通す（関所）

NOTE: このファイルは MAPPER として登録済み（lint_imports.py）。
      型定義の橋渡し（Bridge）+ ドメイン転送の関所という責任を持つ。
"""
from beartype import beartype
from jaxtyping import Float64, UInt32
import jax
from jax import Array
import jax.numpy as jnp
import numpy as np

# ==========================================
# 型エイリアス定義
# ==========================================
NpF64 = Float64[np.ndarray, "..."]
JaxF64 = Float64[Array, "..."]
JaxKey = UInt32[Array, "..."]  # JAX PRNG key (uint32)


# ==========================================
# Domain Gateway（関所）— CPU ↔ GPU 転送
# ==========================================

#takes only NpF64 and returns JaxF64, checks for NaN/Inf, and uses jax.device_put to ensure it's on GPU
@beartype
def to_jax_f64(x: NpF64) -> JaxF64:
    """NumPy(CPU) → JAX(GPU) 変換の関所。

    - beartype が NpF64 (numpy.float64) のみ受け付ける
    - NaN/Inf が混入していたら即クラッシュ
    - jax.device_put で明示的にGPUへ転送
    """
    if np.any(np.isnan(x)):
        raise ValueError(f"NaN detected at CPU→GPU boundary! shape={x.shape}")
    if np.any(np.isinf(x)):
        raise ValueError(f"Inf detected at CPU→GPU boundary! shape={x.shape}")
    ret = jax.device_put(jnp.array(x, dtype=jnp.float64))
    if ret.dtype != jnp.float64:
        print(f"DEBUG to_jax_f64: {x.dtype} -> {ret.dtype}, config={jax.config.read('jax_enable_x64')}")
    return ret

#takes only JaxF64 and returns NpF64, checks for NaN/Inf, and uses np.asarray to ensure it's on CPU
@beartype
def to_np_f64(x: JaxF64) -> NpF64:
    """JAX(GPU) → NumPy(CPU) 変換の関所。

    - beartype が JaxF64 (jax.Array float64) のみ受け付ける
    - NaN/Inf が混入していたら即クラッシュ
    - np.asarray で明示的にCPUへ回収
    """
    result = np.asarray(x, dtype=np.float64)
    if np.any(np.isnan(result)):
        raise ValueError(f"NaN detected at GPU→CPU boundary! shape={result.shape}")
    if np.any(np.isinf(result)):
        raise ValueError(f"Inf detected at GPU→CPU boundary! shape={result.shape}")
    return result
