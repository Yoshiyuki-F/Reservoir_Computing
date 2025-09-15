"""
Reservoir Computing experiment runner module.

This module contains the core experiment execution logic for running
reservoir computing experiments from configuration files.
"""

from typing import Optional, Tuple
import numpy as np

from reservoir import (
    ReservoirComputer,
    ExperimentConfig,
    plot_prediction_results,
    calculate_mse,
    calculate_mae,
    normalize_data,
    denormalize_data
)
from reservoir.data import generate_sine_data, generate_lorenz_data, generate_mackey_glass_data


def run_experiment_from_config(config_path: str, backend: Optional[str] = None) -> Tuple[
    Optional[float], float, Optional[float], float]:
    """設定ファイルからパラメータを読み込んで実験を実行する統合関数。
    
    Args:
        config_path: 設定ファイルのパス
        backend: 明示的に指定するバックエンド ('cpu', 'gpu', または None)
    
    Returns:
        tuple: (train_mse, test_mse, train_mae, test_mae)
    """
    # 設定ファイルから全パラメータを読み込み
    demo_config = ExperimentConfig.from_json(config_path)

    # データ生成関数を選択
    data_generators = {
        'sine_wave': generate_sine_data,
        'lorenz': generate_lorenz_data,
        'mackey_glass': generate_mackey_glass_data
    }

    data_generator_name = demo_config.data_generation.name
    if data_generator_name not in data_generators:
        raise ValueError(f"Unknown data generator: {data_generator_name}")

    data_generator = data_generators[data_generator_name]

    # データ生成（configオブジェクトを直接渡す）
    input_data, target_data = data_generator(demo_config.data_generation)

    # use_dimensionsが指定されている場合は次元を制限
    use_dimensions = demo_config.data_generation.use_dimensions
    if use_dimensions is not None:
        input_data = input_data[:, use_dimensions]
        target_data = target_data[:, use_dimensions]

    print(f"=== {demo_config.demo.title} ===")

    # データを正規化
    input_norm, _, _ = normalize_data(input_data)
    target_norm, target_mean, target_std = normalize_data(target_data)

    # 訓練データとテストデータに分割
    train_size = int(len(input_norm) * demo_config.training.train_size)
    train_input = input_norm[:train_size]
    train_target = target_norm[:train_size]
    test_input = input_norm[train_size:]
    test_target = target_norm[train_size:]

    # Reservoir Computerを初期化
    rc = ReservoirComputer(config=demo_config.reservoir, backend=backend)

    print(f"Reservoir情報: {rc.get_reservoir_info()}")

    # 訓練
    print("訓練中...")
    rc.train(train_input, train_target, reg_param=demo_config.training.reg_param)

    # 予測
    print("予測中...")
    test_predictions = rc.predict(test_input)

    # 正規化を元に戻す
    test_predictions = denormalize_data(test_predictions, target_mean, target_std)
    test_target_orig = denormalize_data(test_target, target_mean, target_std)

    # 誤差計算
    test_mse = calculate_mse(test_predictions, test_target_orig)
    test_mae = calculate_mae(test_predictions, test_target_orig)

    train_mse = train_mae = None
    train_predictions_orig = train_target_orig = None

    if demo_config.demo.show_training:
        train_predictions = rc.predict(train_input)
        train_predictions_orig = denormalize_data(train_predictions, target_mean, target_std)
        train_target_orig = denormalize_data(train_target, target_mean, target_std)

        train_mse = calculate_mse(train_predictions_orig, train_target_orig)
        train_mae = calculate_mae(train_predictions_orig, train_target_orig)

        print(f"訓練 MSE: {train_mse:.6f}, MAE: {train_mae:.6f}")

    print(f"テスト MSE: {test_mse:.6f}, MAE: {test_mae:.6f}")

    # 可視化
    if demo_config.demo.show_training:
        plot_prediction_results(
            test_target_orig, test_predictions, np.arange(len(test_target_orig)),
            demo_config.demo.title, demo_config.demo.filename,
            train_target_orig, train_predictions_orig, train_size
        )
    else:
        plot_prediction_results(
            test_target_orig, test_predictions, np.arange(len(test_target_orig)),
            demo_config.demo.title, demo_config.demo.filename
        )

    return train_mse, test_mse, train_mae, test_mae
