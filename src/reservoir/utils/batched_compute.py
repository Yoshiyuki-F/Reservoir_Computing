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
        inputs: 入力データ (numpy array) - 2D (T, F) or 3D (N, T, F)
        batch_size: バッチサイズ
        desc: tqdm進捗表示のラベル

    Returns:
        出力データ (numpy array)
    """
    inputs_np = np.asarray(inputs)
    
    # Handle 2D input (T, F) - Regression time series
    # Process entire sequence at once (no batching along time axis)
    if inputs_np.ndim == 2:
        result_jax = fn(jnp.array(inputs_np))
        return np.asarray(result_jax, dtype=np.float32)
    
    # 3D input (N, T, F) - Classification batching
    n_samples = inputs_np.shape[0]

    if n_samples == 0:
        return np.array([])

    # 1. 形状推論 & JITコンパイルのトリガー (最初の1サンプル)
    dummy_input_jax = jnp.array(inputs_np[:1])
    dummy_out_jax = fn(dummy_input_jax)

    # Detect Expansion Factor (e.g. 1 sample -> N samples after aggregation)
    dummy_in_size = dummy_input_jax.shape[0]
    dummy_out_size = dummy_out_jax.shape[0]
    
    expansion_factor = 1
    if dummy_in_size > 0:
        expansion_factor = dummy_out_size // dummy_in_size

    # Calculate Total Output Size
    total_output_samples = n_samples * expansion_factor
    output_shape = (total_output_samples,) + dummy_out_jax.shape[1:]

    # 2. CPU側に結果格納用のメモリを確保 (float32でメモリ節約)
    output = np.empty(output_shape, dtype=np.float32)

    # 3. JITコンパイル済みの実行関数を用意
    @jax.jit
    def step(x):
        return fn(x)

    # 4. バッチ処理ループ (tqdm適用)
    # Output Index Tracker
    out_idx = 0
    
    with tqdm(total=n_samples, desc=desc, unit="samples") as pbar:
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            current_batch_size = batch_end - i

            # (A) CPUでスライス
            batch_np = inputs_np[i:batch_end]

            # (B) GPUへ転送 -> 計算
            batch_out_jax = step(jnp.array(batch_np))
            
            # (C) CPUへ戻す (同期)
            # Map input batch size to output batch size
            current_out_size = current_batch_size * expansion_factor
            
            # Ensure shape match (handle partial batches or simple mismatches)
            batch_result_np = np.asarray(batch_out_jax)
            
            # If flattening happens (e.g. (B, T, F) -> (B*T, F)), shape[0] is B*T
            # So we just fill the buffer sequentially
            output[out_idx : out_idx + current_out_size] = batch_result_np
            
            out_idx += current_out_size

            # 進捗更新
            pbar.update(current_batch_size)

    return output
