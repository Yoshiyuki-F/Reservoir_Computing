"""
utils/batched_compute.py
GPU OOMを防ぐためのバッチ処理ユーティリティ。
"""

from typing import Callable, Union, Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm


def batched_compute(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    inputs: Union[np.ndarray, jnp.ndarray, Any],
    batch_size: int,
    desc: str = "[Batched]",
) -> np.ndarray:
    """
    データセット全体を一括でGPUに載せるとOOMになるため、
    バッチごとにJAX(GPU)で計算し、結果をCPU(Numpy)に退避させる関数。

    Args:
        fn: JAX関数（projection, feature extractionなど）
        inputs: 入力データ - 2D (T, F) or 3D (N, T, F)
        batch_size: バッチサイズ
        desc: tqdm進捗表示のラベル

    Returns:
        出力データ (numpy array on CPU)
    """
    # Do NOT force inputs to GPU/JAX immediately if it's numpy
    # inputs_jax = jnp.asarray(inputs) 
    
    # Handle 2D input (T, F) - Regression time series
    # Check ndim on inputs (works for both np and jnp)
    if inputs.ndim == 2:
        inputs_jax = jnp.asarray(inputs)
        result_jax = fn(inputs_jax)
        return np.asarray(result_jax)  # Transfer to CPU
    
    # 3D input (N, T, F) - Classification batching
    n_samples = inputs.shape[0]

    if n_samples == 0:
        return np.array([])

    # 1. 形状推論 & JITコンパイルのトリガー (最初の1サンプル)
    # Ensure dummy input is on GPU
    dummy_input_jax = jnp.asarray(inputs[:1])
    dummy_out_jax = fn(dummy_input_jax)

    # Detect Expansion Factor (e.g. 1 sample -> N samples after aggregation)
    dummy_in_size = dummy_input_jax.shape[0]
    dummy_out_size = dummy_out_jax.shape[0]
    
    expansion_factor = 1
    if dummy_in_size > 0:
        expansion_factor = dummy_out_size // dummy_in_size

    # 2. Collect results on CPU (numpy list)
    results = []

    # 3. JITコンパイル済みの実行関数を用意
    @jax.jit
    def step(x):
        return fn(x)

    # 4. バッチ処理ループ (tqdm適用)
    with tqdm(total=n_samples, desc=desc, unit="samples") as pbar:
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            current_batch_size = batch_end - i

            # (A) GPUでスライス & 計算
            batch_data = inputs[i:batch_end]
            batch_jax = jnp.asarray(batch_data)
            batch_out_jax = step(batch_jax)
            
            # (B) Transfer to CPU immediately to free GPU memory
            results.append(np.asarray(batch_out_jax))

            # 進捗更新
            pbar.update(current_batch_size)

    # Concatenate all results on CPU
    return np.concatenate(results, axis=0)
