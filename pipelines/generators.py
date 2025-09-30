"""
Reservoir Computing用のデータ生成関数。
"""

from typing import Tuple

import numpy as np

from .jax_config import ensure_x64_enabled

ensure_x64_enabled()

import jax.numpy as jnp

from configs.core import DataGenerationConfig


def generate_sine_data(config: DataGenerationConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """複数周波数のサイン波を合成した時系列データを生成。
    
    複数の正弦波を重ね合わせて合成信号を作成し、ガウシアンノイズを
    追加した時系列データを生成します。Reservoir Computingの訓練・
    テスト用データとして使用されます。
    
    Args:
        config: DataGenerationConfig オブジェクト
            
    Returns:
        tuple: (入力データ, 目標データ) のペア
            - 入力データ: 形状 (time_steps-1, 1) の現在値
            - 目標データ: 形状 (time_steps-1, 1) の次時刻値
            
    Examples:
        単一周波数のサイン波:
        
        >>> from configs.core import DataGenerationConfig
        >>> config = DataGenerationConfig(
        ...     name="sine_wave",
        ...     time_steps=1000,
        ...     dt=0.01,
        ...     noise_level=0.05,
        ...     params={"frequencies": [1.0]}
        ... )
        >>> inputs, targets = generate_sine_data(config)
        >>> print(inputs.shape)
        (999, 1)
    """
    t = np.arange(config.time_steps, dtype=np.float64) * config.dt
    
    # 複数の周波数のサインwave合成
    signal = np.zeros(config.time_steps, dtype=np.float64)
    frequencies = config.get_param('frequencies')
    if not frequencies:
        raise ValueError("sine_wave requires 'frequencies' parameter in config.params")
    
    for freq in frequencies:
        signal += np.sin(2 * np.pi * freq * t)
    
    # ノイズを追加
    noise = np.random.normal(0, config.noise_level, config.time_steps).astype(np.float64)
    signal += noise
    
    # 入力は現在の値、ターゲットは次の値
    input_data = jnp.array(signal[:-1].reshape(-1, 1))   # 形状: (time_steps-1, 1)
    target_data = jnp.array(signal[1:].reshape(-1, 1))   # 形状: (time_steps-1, 1)
    
    return input_data, target_data


def generate_lorenz_data(config: DataGenerationConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Lorenz方程式による決定論的カオス時系列データを生成。
    
    Lorenzアトラクターは気象学から生まれた有名なカオス系で、
    決定論的でありながら複雑で予測困難な時系列を生成します。
    Reservoir Computingのテストに適した非線形動力学系です。
    
    微分方程式系:
        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y  
        dz/dt = xy - βz
        
    Args:
        config: DataGenerationConfig オブジェクト
        
    Returns:
        tuple: (入力データ, 目標データ) のペア
            - 入力データ: 形状 (time_steps-1, 3) の現在状態 [x, y, z]
            - 目標データ: 形状 (time_steps-1, 3) の次時刻状態
            
    Examples:
        標準的なLorenzパラメータでカオス時系列生成:
        
        >>> from configs.core import DataGenerationConfig
        >>> config = DataGenerationConfig(
        ...     name="lorenz",
        ...     time_steps=5000,
        ...     dt=0.01,
        ...     noise_level=0.01,
        ...     params={
        ...         "sigma": 10.0,
        ...         "rho": 28.0,
        ...         "beta": 8.0/3.0
        ...     }
        ... )
        >>> inputs, targets = generate_lorenz_data(config)
        >>> print(inputs.shape)
        (4999, 3)
        
    Note:
        数値積分にはオイラー法を使用。より高精度が必要な場合は
        Runge-Kutta法の実装を検討してください。
    """
    # 初期値（互換性のためオプショナル）
    x = config.get_param('initial_x', 1.0)
    y = config.get_param('initial_y', 1.0) 
    z = config.get_param('initial_z', 1.0)
    
    data = np.zeros((config.time_steps, 3), dtype=np.float64)
    
    for i in range(config.time_steps):
        # Lorenz方程式の数値積分（オイラー法）
        sigma = config.get_param('sigma')
        rho = config.get_param('rho')
        beta = config.get_param('beta')
        
        if sigma is None or rho is None or beta is None:
            raise ValueError("lorenz requires 'sigma', 'rho', 'beta' parameters in config.params")
            
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        x += dx * config.dt
        y += dy * config.dt
        z += dz * config.dt
        
        data[i] = [x, y, z]
    
    # 入力は現在の状態、ターゲットは次の状態
    input_data = jnp.array(data[:-1], dtype=jnp.float64)
    target_data = jnp.array(data[1:], dtype=jnp.float64)
    
    return input_data, target_data


def generate_mackey_glass_data(config: DataGenerationConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Mackey-Glassカオス時系列データを生成します。
    
    Args:
        config: DataGenerationConfig オブジェクト
        
    Returns:
        (input_data, target_data): 入力データとターゲットデータのタプル
        
    Note:
        Mackey-Glass方程式の標準的なパラメータ:
        tau=17, beta=0.2, gamma=0.1, n=10
        これらはconfigに含まれていない場合のデフォルト値として使用されます。
    """
    # Mackey-Glassのパラメータ（configから取得）
    tau = config.get_param('tau')
    beta = config.get_param('beta')
    gamma = config.get_param('gamma')
    n = config.get_param('n')
    
    if any(param is None for param in [tau, beta, gamma, n]):
        raise ValueError("mackey_glass requires 'tau', 'beta', 'gamma', 'n' parameters in config.params")
    
    # 初期化
    history_length = max(int(tau), 1) + 1
    x = np.zeros(config.time_steps + history_length, dtype=np.float64)
    initial_value = config.get_param('initial_value', 1.2)
    x[0] = initial_value
    
    for i in range(history_length, config.time_steps + history_length):
        x_tau = x[i - int(tau)] if i >= int(tau) else x[0]
        dx = (beta * x_tau) / (1 + x_tau**n) - gamma * x[i-1]
        x[i] = x[i-1] + dx * config.dt
    
    # 履歴部分を除去
    x = x[history_length:]
    
    # ノイズを追加
    if config.noise_level > 0:
        noise = np.random.normal(0, config.noise_level, len(x)).astype(np.float64)
        x += noise
    
    # 入力は現在の値、ターゲットは次の値
    input_data = jnp.array(x[:-1].reshape(-1, 1), dtype=jnp.float64)
    target_data = jnp.array(x[1:].reshape(-1, 1), dtype=jnp.float64)
    
    return input_data, target_data
