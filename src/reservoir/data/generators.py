"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/generators.py
Reservoir Computing用のデータ生成関数。
"""

from typing import Tuple, Optional

import numpy as np
import jax.numpy as jnp

try:
    import torch  # type: ignore
    from reservoir.data.mnist_loader import get_mnist_datasets, image_to_sequence
except ModuleNotFoundError:  # pragma: no cover - torch optional
    torch = None
    get_mnist_datasets = None  # type: ignore
    image_to_sequence = None  # type: ignore

from reservoir.utils import ensure_x64_enabled

ensure_x64_enabled()

from reservoir.data.config import DataGenerationConfig


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


def generate_mnist_sequence_data(
    config: DataGenerationConfig,
    *,
    split: Optional[str] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate MNIST-based sequence data by scanning images as time series.

    Args:
        config: DataGenerationConfig with params such as:
            - split: str, 'train' or 'test' (default 'train')
            - train_fraction/test_fraction/fraction: optional floats in (0, 1]
        and fields:
            - time_steps: Number of time steps to reshape each image into.

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

    split_name = split or config.get_param("split", "train")

    total_pixels = 28 * 28
    time_steps = getattr(config, "time_steps", None)
    if time_steps is None:
        raise ValueError("MNIST configuration must specify time_steps")
    n_steps = int(time_steps)
    if n_steps <= 0:
        raise ValueError(f"time_steps must be positive, got {n_steps}")
    if total_pixels % n_steps != 0:
        raise ValueError(
            f"time_steps={n_steps} must evenly divide {total_pixels} to reshape MNIST images"
        )

    train_set, test_set = get_mnist_datasets()
    if split_name not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got '{split_name}'")
    dataset = train_set if split_name == "train" else test_set

    max_available = len(dataset)
    fraction_key = f"{split_name}_fraction"
    fraction_param = config.get_param(fraction_key)
    if fraction_param is None:
        fraction_param = config.get_param("fraction")

    if fraction_param is None:
        limit = max_available
    else:
        fraction_val = float(fraction_param)
        if fraction_val <= 0 or fraction_val > 1:
            raise ValueError(f"{fraction_key if config.get_param(fraction_key) is not None else 'fraction'} must be in (0, 1], got {fraction_val}")
        limit = max(1, min(max_available, int(max_available * fraction_val)))

    sequences = []
    labels = []

    for idx in range(limit):
        img_tensor, label = dataset[idx]
        img_np = img_tensor.numpy()
        seq = image_to_sequence(img_np, n_steps=n_steps)
        sequences.append(seq.astype(np.float64))
        labels.append(label)

    input_data = jnp.array(sequences, dtype=jnp.float64)
    target_labels = jnp.array(labels, dtype=jnp.int32)
    return input_data, target_labels
