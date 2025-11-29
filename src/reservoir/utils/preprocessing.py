"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/utils/preprocessing.py
Reservoir Computing用のデータ前処理関数。
"""

from typing import Tuple

import numpy as np

from .jax_config import ensure_x64_enabled

ensure_x64_enabled()

import jax.numpy as jnp


def normalize_data(data: jnp.ndarray) -> Tuple[jnp.ndarray, float, float]:
    """データを標準化（平均0、標準偏差1に変換）。
    
    時系列データを安定して学習させるために、各特徴量を
    標準正規分布に従うように変換します。逆変換（denormalize）に
    必要な平均と標準偏差も返します。
    
    Args:
        data: 正規化するデータ配列
        
    Returns:
        tuple: (正規化済みデータ, 元の平均, 元の標準偏差)
        
    Examples:
        >>> data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> normalized, mean, std = normalize_data(data)
        >>> print(normalized)
        [-1.41421356 -0.70710678  0.          0.70710678  1.41421356]
        >>> print(f"{mean=}, {std=:.2f}")
        mean=3.0, std=1.41
    """
    # float64精度で計算
    data = data.astype(jnp.float64)
    mean = jnp.mean(data)
    std = jnp.std(data)
    # 標準偏差が0に近い場合の対策
    std = jnp.maximum(std, 1e-12)
    normalized_data = (data - mean) / std
    return normalized_data, float(mean), float(std)


def denormalize_data(normalized_data: jnp.ndarray, mean: float, std: float) -> jnp.ndarray:
    """正規化されたデータを元のスケールに復元。
    
    `normalize_data` で変換されたデータを、元の平均と標準偏差を
    使って元のスケールに戻します。予測結果を解釈する際に使用します。
    
    Args:
        normalized_data: 正規化されたデータ配列
        mean: 元のデータの平均値
        std: 元のデータの標準偏差
        
    Returns:
        元のスケールに復元されたデータ配列
        
    Examples:
        >>> normalized = jnp.array([-1.414, -0.707, 0.0, 0.707, 1.414])
        >>> mean, std = 3.0, 1.414
        >>> original = denormalize_data(normalized, mean, std)
        >>> print(original)
        [1. 2. 3. 4. 5.]
    """
    normalized_data = normalized_data.astype(jnp.float64)
    return normalized_data * std + mean


def create_sliding_windows(
    input_data: jnp.ndarray,
    target_data: jnp.ndarray,
    window_size: int,
    horizon: int = 1,
    stride: int = 1,
    target: str = "next",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate sliding-window datasets for sequence prediction."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    if input_data.ndim != 2 or target_data.ndim != 2:
        raise ValueError("input_data and target_data must be 2D arrays")

    if input_data.shape[0] != target_data.shape[0]:
        raise ValueError("input_data and target_data must share the same number of timesteps")

    total_steps = input_data.shape[0]
    max_start = total_steps - (window_size + horizon - 1)
    if max_start < 0:
        raise ValueError(
            "window_size and horizon exceed available timesteps"
        )

    if target not in {"next", "last"}:
        raise ValueError("target must be either 'next' or 'last'")

    windows: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for start in range(0, max_start + 1, stride):
        end = start + window_size
        window_slice = np.asarray(input_data[start:end])

        if target == "next":
            target_index = end - 1 + (horizon - 1)
        else:  # target == "last"
            target_index = end - 1

        window_target = np.asarray(target_data[target_index])

        windows.append(window_slice.reshape(-1))
        targets.append(window_target)

    window_array = jnp.array(windows, dtype=jnp.float64)
    target_array = jnp.array(targets, dtype=jnp.float64)

    return window_array, target_array
