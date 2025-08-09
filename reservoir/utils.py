"""
Reservoir Computing用のユーティリティ関数。
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple


def generate_sine_data(
    time_steps: int = 1000,
    dt: float = 0.01,
    frequencies: list = [1.0, 2.0],
    noise_level: float = 0.05
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    サイン波データを生成します。
    
    Args:
        time_steps: 時系列の長さ
        dt: 時間ステップ
        frequencies: サイン波の周波数リスト
        noise_level: ノイズレベル
        
    Returns:
        (input_data, target_data): 入力データとターゲットデータのタプル
    """
    t = np.arange(time_steps, dtype=np.float64) * dt
    
    # 複数の周波数のサインwave合成
    signal = np.zeros(time_steps, dtype=np.float64)
    for freq in frequencies:
        signal += np.sin(2 * np.pi * freq * t)
    
    # ノイズを追加
    noise = np.random.normal(0, noise_level, time_steps).astype(np.float64)
    signal += noise
    
    # 入力は現在の値、ターゲットは次の値
    input_data = jnp.array(signal[:-1].reshape(-1, 1))   # 形状: (time_steps-1, 1)
    target_data = jnp.array(signal[1:].reshape(-1, 1))   # 形状: (time_steps-1, 1)
    
    return input_data, target_data


def generate_lorenz_data(
    time_steps: int = 1000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0/3.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Lorenzアトラクターデータを生成します。
    
    Args:
        time_steps: 時系列の長さ
        dt: 時間ステップ
        sigma, rho, beta: Lorenzアトラクターのパラメータ
        
    Returns:
        (input_data, target_data): 入力データとターゲットデータのタプル
    """
    # 初期値
    x, y, z = 1.0, 1.0, 1.0
    
    data = np.zeros((time_steps, 3), dtype=np.float64)
    
    for i in range(time_steps):
        # Lorenz方程式の数値積分（オイラー法）
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        x += dx * dt
        y += dy * dt
        z += dz * dt
        
        data[i] = [x, y, z]
    
    # 入力は現在の状態、ターゲットは次の状態
    input_data = jnp.array(data[:-1], dtype=jnp.float64)
    target_data = jnp.array(data[1:], dtype=jnp.float64)
    
    return input_data, target_data


def generate_mackey_glass_data(
    time_steps: int = 1000,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: int = 10,
    dt: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Mackey-Glassカオス時系列データを生成します。
    
    Args:
        time_steps: 時系列の長さ
        tau: 遅延時間
        beta, gamma, n: Mackey-Glass方程式のパラメータ
        dt: 時間ステップ
        
    Returns:
        (input_data, target_data): 入力データとターゲットデータのタプル
    """
    # 初期化
    history_length = max(tau, 1) + 1
    x = np.zeros(time_steps + history_length, dtype=np.float64)
    x[0] = 1.2  # 初期値
    
    for i in range(history_length, time_steps + history_length):
        x_tau = x[i - tau] if i >= tau else x[0]
        dx = (beta * x_tau) / (1 + x_tau**n) - gamma * x[i-1]
        x[i] = x[i-1] + dx * dt
    
    # 履歴部分を除去
    x = x[history_length:]
    
    # 入力は現在の値、ターゲットは次の値
    input_data = jnp.array(x[:-1].reshape(-1, 1), dtype=jnp.float64)
    target_data = jnp.array(x[1:].reshape(-1, 1), dtype=jnp.float64)
    
    return input_data, target_data


def normalize_data(data: jnp.ndarray) -> Tuple[jnp.ndarray, float, float]:
    """
    データを正規化します。
    
    Args:
        data: 正規化するデータ
        
    Returns:
        (normalized_data, mean, std): 正規化されたデータ、平均、標準偏差
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
    """
    正規化されたデータを元のスケールに戻します。
    
    Args:
        normalized_data: 正規化されたデータ
        mean: 元のデータの平均
        std: 元のデータの標準偏差
        
    Returns:
        denormalized_data: 元のスケールのデータ
    """
    normalized_data = normalized_data.astype(jnp.float64)
    return normalized_data * std + mean


def calculate_mse(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
    """
    平均二乗誤差を計算します。
    
    Args:
        predictions: 予測値
        targets: 正解値
        
    Returns:
        mse: 平均二乗誤差
    """
    predictions = predictions.astype(jnp.float64)
    targets = targets.astype(jnp.float64)
    return float(jnp.mean((predictions - targets) ** 2))


def calculate_mae(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
    """
    平均絶対誤差を計算します。
    
    Args:
        predictions: 予測値
        targets: 正解値
        
    Returns:
        mae: 平均絶対誤差
    """
    predictions = predictions.astype(jnp.float64)
    targets = targets.astype(jnp.float64)
    return float(jnp.mean(jnp.abs(predictions - targets))) 