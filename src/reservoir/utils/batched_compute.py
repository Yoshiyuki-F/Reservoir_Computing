"""
utils/batched_compute.py
GPU OOMを防ぐためのバッチ処理ユーティリティ。
"""

from collections.abc import Callable
from beartype import beartype
from reservoir.core.types import NpF64, JaxF64, to_jax_f64, to_np_f64
from reservoir.utils.reporting import print_feature_stats

import jax
import numpy as np
from tqdm.auto import tqdm


@beartype
def batched_compute(
    fn: Callable[[JaxF64], JaxF64],
    inputs: NpF64,
    batch_size: int,
    desc: str = "[Batched]",
) -> NpF64:
    """
    データセット全体を一括でGPUに載せるとOOMになるため、
    バッチごとにJAX(GPU)で計算し、結果をCPU(Numpy)に退避させる関数。
    tqdm + GPUバッチ処理のオーケストレーションを提供。

    Args:
        fn: JAX関数（projection, feature extractionなど）
        inputs: 入力データ - 2D (T, F) or 3D (N, T, F)
        batch_size: バッチサイズ
        desc: tqdm進捗表示のラベル

    Returns:
        出力データ (numpy array on CPU)
    """
    
    # Handle 2D input (T, F) - Regression time series
    # Check ndim on inputs (works for both np and jnp)
    if inputs.ndim == 2:
        inputs_jax = to_jax_f64(inputs)
        result_jax = fn(inputs_jax)
        return to_np_f64(result_jax)  # Transfer to CPU
    
    # 3D input (N, T, F) - Classification batching
    n_samples = inputs.shape[0]

    if n_samples == 0:
        return np.array([])

    # 1. 形状推論 & JITコンパイルのトリガー (最初の1サンプル)
    # Ensure dummy input is on GPU and is float64
    if not jax.config.read("jax_enable_x64"):
        raise ValueError("JAX is not configured for float64! Please enable jax_enable_x64 for this function.")
        
    dummy_input_jax = to_jax_f64(inputs[:1])
    dummy_out_jax = fn(dummy_input_jax)

    # Detect Expansion Factor (e.g. 1 sample -> N samples after aggregation)
    dummy_in_size = dummy_input_jax.shape[0]
    dummy_out_size = dummy_out_jax.shape[0]
    
    expansion_factor = 1
    if dummy_in_size > 0:
        expansion_factor = dummy_out_size // dummy_in_size

    # 2. Pre-allocate results array on CPU to prevent OOM
    out_shape = (n_samples * expansion_factor, *dummy_out_jax.shape[1:])
    result_array = np.empty(out_shape, dtype=np.float64)

    # 3. JITコンパイル済みの実行関数を用意
    @jax.jit
    def step(x: JaxF64) -> JaxF64:
        return fn(x)

    # 4. バッチ処理ループ (tqdm適用)
    with tqdm(total=n_samples, desc=desc, unit="samples") as pbar:
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            current_batch_size = batch_end - i

            # (A) GPUでスライス & 計算
            batch_data = inputs[i:batch_end]
            batch_jax = to_jax_f64(batch_data)
            batch_out_jax = step(batch_jax)
            
            # (B) Transfer to CPU directly into pre-allocated array
            out_start = i * expansion_factor
            out_end = out_start + batch_out_jax.shape[0]
            result_array[out_start:out_end] = to_np_f64(batch_out_jax)

            # 進捗更新
            pbar.update(current_batch_size)

    print_feature_stats(result_array, desc+" Output")
    return result_array
