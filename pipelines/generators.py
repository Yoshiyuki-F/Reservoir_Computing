"""
Reservoir Computing用のデータ生成関数。
"""

from typing import Tuple, Optional

import numpy as np
import jax.numpy as jnp

try:
    import torch
    from pipelines.datasets.mnist_loader import (
        get_mnist_datasets,
        image_to_sequence,
    )
except ModuleNotFoundError:  # pragma: no cover - torch optional
    torch = None
    get_mnist_datasets = None  # type: ignore
    image_to_sequence = None  # type: ignore

from .jax_config import ensure_x64_enabled

ensure_x64_enabled()

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
        トランジェント除去が不要な場合は`warmup_steps`を省略（または0）できます。
        tauは連続時間の遅延として解釈され、離散化ステップdtで割ることでサンプル遅延数に変換されます。
    """
    # Mackey-Glassのパラメータ（configから取得）
    tau = config.get_param('tau')
    beta = config.get_param('beta')
    gamma = config.get_param('gamma')
    n = config.get_param('n')

    if any(param is None for param in [tau, beta, gamma, n]):
        raise ValueError("mackey_glass requires 'tau', 'beta', 'gamma', 'n' parameters in config.params")

    dt = float(config.dt)
    if dt <= 0:
        raise ValueError("dt must be positive for Mackey-Glass generation")

    # tauの単位をdtに合わせた遅延ステップ数へ変換
    delay_steps = int(np.round(float(tau) / dt))
    if delay_steps < 1:
        delay_steps = 1

    # 初期化とトランジェント除去用のウォームアップ（省略可）
    history_length = delay_steps + 1
    warmup_source = config.warmup_steps
    if warmup_source is None:
        warmup_source = config.get_param('warmup_steps', 0)

    warmup_steps = max(int(warmup_source), 0)
    total_steps = config.time_steps + history_length + warmup_steps

    x = np.full(total_steps, fill_value=0.0, dtype=np.float64)
    initial_value = config.get_param('initial_value', 1.2)
    x[:history_length] = initial_value

    for i in range(history_length, total_steps):
        x_prev = x[i - 1]
        x_tau = x[i - delay_steps]
        dx = (beta * x_tau) / (1 + x_tau**n) - gamma * x_prev
        x[i] = x_prev + dt * dx

    # トランジェント除去
    start_idx = history_length + warmup_steps
    end_idx = start_idx + config.time_steps
    if end_idx > len(x):
        raise ValueError("Warmup and history exceed generated sequence length")
    x = x[start_idx:end_idx]

    # ノイズを追加
    if config.noise_level > 0:
        noise = np.random.normal(0, config.noise_level, len(x)).astype(np.float64)
        x += noise

    # 入力は現在の値、ターゲットは次の値
    input_data = jnp.array(x[:-1].reshape(-1, 1), dtype=jnp.float64)
    target_data = jnp.array(x[1:].reshape(-1, 1), dtype=jnp.float64)

    return input_data, target_data


def generate_mnist_sequence_data(config: DataGenerationConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate MNIST-based sequence data by scanning images as time series.

    Args:
        config: DataGenerationConfig with params:
            - limit: Optional[int], number of samples to load (default 1000)
            - split: str, 'train' or 'test' (default 'train')
            - encoding: str, 'cols' or 'flat' (default 'cols')

    Returns:
        Tuple (input_sequences, labels) where sequences are float64 JAX arrays.

    Raises:
        ImportError: If torch/torchvision are not available.
    """
    if get_mnist_datasets is None or image_to_sequence is None or torch is None:
        raise ImportError(
            "MNIST generation requires torch and torchvision. "
            "Install them to enable this feature."
        )

    limit = config.get_param("limit")
    split = config.get_param("split", "train")
    encoding = config.get_param("sequence_encoding", config.get_param("encoding", "cols"))

    train_set, test_set = get_mnist_datasets()
    dataset = train_set if split == "train" else test_set

    max_available = len(dataset)
    if limit is None:
        limit = max_available
    else:
        limit = int(limit)
        if limit < 0:
            raise ValueError("limit must be non-negative")
        limit = min(limit, max_available)

    sequences = []
    labels = []

    for idx in range(limit):
        img_tensor, label = dataset[idx]
        img_np = img_tensor.numpy()
        seq = image_to_sequence(img_np, mode=encoding)
        sequences.append(seq.astype(np.float64))
        labels.append(label)

    input_data = jnp.array(sequences, dtype=jnp.float64)
    target_labels = jnp.array(labels, dtype=jnp.int32)
    return input_data, target_labels
