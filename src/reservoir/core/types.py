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
from jaxtyping import Float64, UInt32, jaxtyped
import jax
from jax import Array
import jax.numpy as jnp
import numpy as np

from typing import TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    import reservoir.core.interfaces
    import reservoir.layers.preprocessing

# ==========================================
# 型エイリアス定義
# ==========================================
NpF64 = Float64[np.ndarray, "..."]
JaxF64 = Float64[Array, "..."]
JaxKey = UInt32[Array, "..."]  # JAX PRNG key (uint32)

class TrainLogs(TypedDict, total=False):
    """Strictly typed training logs to replace Dict[str, object]."""
    loss_history: list[float]
    final_loss: float
    distill_mse: float
    accuracy: float
    # Add other specific keys as they emerge.

class EvalMetrics(TypedDict, total=False):
    """Strictly typed evaluation metrics to replace Dict[str, float]."""
    mse: float
    mae: float
    accuracy: float
    # Chaos Metrics
    nmse: float
    nrmse: float
    mase: float
    ndei: float
    var_ratio: float
    correlation: float
    vpt_steps: float 
    vpt_lt: float
    vpt_threshold: float
    # Add other specific keys as they emerge.

class TrainMetrics(EvalMetrics, total=False):
    weight_norms: dict[float, float]

class TestMetrics(EvalMetrics, total=False):
    pass

# ==========================================
# Config Domain Types (Nesting, No Recursion to satisfy beartype)
# ==========================================

# 値になりうる基本型
PrimitiveValue = str | float | int | bool | None

# ネストを階層的に定義 (L1 -> L2 -> L3)
# L1: 基本型とそのコレクション
ConfigL1 = PrimitiveValue | tuple[PrimitiveValue, ...] | list[PrimitiveValue] | dict[str, PrimitiveValue]
# L2: L1を含むコレクション (DistillationConfigなどで使用)
ConfigL2 = ConfigL1 | tuple[ConfigL1, ...] | list[ConfigL1] | dict[str, ConfigL1]
# L3: L2を含むコレクション (将来用)
ConfigL3 = ConfigL2 | tuple[ConfigL2, ...] | list[ConfigL2] | dict[str, ConfigL2]

# 全ての to_dict() の戻り値
ConfigDict = dict[str, ConfigL3]
ConfigValue = ConfigL3

# ==========================================
# Result Domain Types (Execution Outputs)
# ==========================================

class FitResultMetrics(TypedDict, total=False):
    train: EvalMetrics
    val: EvalMetrics
    test: EvalMetrics

class FitResultDict(TypedDict, total=False):
    train_pred: JaxF64 | None
    val_pred: JaxF64 | None
    test_pred: JaxF64 | None
    metrics: FitResultMetrics
    best_lambda: float | None
    best_score: float | None
    search_history: dict[float, float]
    weight_norms: dict[float, float]
    residuals_history: dict[float, NpF64] | None
    closed_loop_pred: JaxF64 | None
    closed_loop_truth: JaxF64 | None
    chaos_results: dict[str, float] | None
    outputs: dict[str, JaxF64 | None]
    aligned_test_y: JaxF64 | None

class ResultDict(TypedDict, total=False):
    fit_result: FitResultDict
    train_logs: TrainLogs
    quantum_trace: JaxF64 | None
    train: TrainMetrics
    test: TestMetrics
    validation: EvalMetrics
    outputs: dict[str, NpF64 | None]
    readout: "reservoir.core.interfaces.ReadoutModule | None"
    preprocessor: "reservoir.layers.preprocessing.Preprocessor | None"
    scaler: "reservoir.layers.preprocessing.Preprocessor | None"
    training_logs: TrainLogs
    meta: dict[str, float | str]
    is_closed_loop: bool
    residuals_history: dict[float, NpF64] | None

# **kwargs 用の厳格な型定義 (No Any)
KwargsDict = dict[str, PrimitiveValue | JaxF64 | NpF64 | tuple[PrimitiveValue, ...] | list[PrimitiveValue] | ConfigDict | ResultDict]


# ==========================================
# Domain Gateway（関所）— CPU ↔ GPU 転送
# ==========================================

#takes only NpF64 and returns JaxF64, checks for NaN/Inf, and uses jax.device_put to ensure it's on GPU
@jaxtyped(typechecker=beartype)
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
@jaxtyped(typechecker=beartype)
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
