"""
utils/batched_compute.py
GPU OOMを防ぐためのバッチ処理ユーティリティ。
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm


def batched_compute(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    inputs: np.ndarray,
    batch_size: int,
    desc: str = "[Batched]",
) -> np.ndarray:
    """
    データセット全体を一括でGPUに載せるとOOMになるため、
    バッチごとにJAX(GPU)で計算し、結果をCPU(Numpy)に退避させる関数。

    Args:
        fn: JAX関数（projection, feature extractionなど）
        inputs: 入力データ (numpy array)
        batch_size: バッチサイズ
        desc: tqdm進捗表示のラベル

    Returns:
        出力データ (numpy array)
    """
    inputs_np = np.asarray(inputs)
    n_samples = inputs_np.shape[0]

    if n_samples == 0:
        return np.array([])

    # 1. 形状推論 & JITコンパイルのトリガー (最初の1サンプル)
    dummy_input_jax = jnp.array(inputs_np[:1])
    dummy_out_jax = fn(dummy_input_jax)

    output_shape = (n_samples,) + dummy_out_jax.shape[1:]

    # 2. CPU側に結果格納用のメモリを確保 (float32でメモリ節約)
    output = np.empty(output_shape, dtype=np.float32)

    # 3. JITコンパイル済みの実行関数を用意
    @jax.jit
    def step(x):
        return fn(x)

    # 4. バッチ処理ループ (tqdm適用)
    with tqdm(total=n_samples, desc=desc, unit="samples") as pbar:
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            current_batch_size = batch_end - i

            # (A) CPUでスライス
            batch_np = inputs_np[i:batch_end]

            # (B) GPUへ転送 -> 計算
            batch_out_jax = step(jnp.array(batch_np))

            # (C) CPUへ戻す (同期)
            output[i:batch_end] = np.asarray(batch_out_jax)

            # 進捗更新
            pbar.update(current_batch_size)

    return output
