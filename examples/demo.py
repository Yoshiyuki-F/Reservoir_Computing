"""
JAXを使ったReservoir Computingのデモンストレーション。

この例では、サイン波の時系列予測を行います。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np

from reservoir import ReservoirComputer, DemoConfig
from reservoir.utils import (
    check_gpu_available,
    generate_sine_data,
    generate_lorenz_data,
    normalize_data,
    denormalize_data,
    calculate_mse,
    calculate_mae
)


def plot_prediction_results(ground_truth, predictions, time_steps, title, filename, 
                          train_data=None, train_predictions=None, train_size=None):
    """予測結果を可視化して保存する共通関数"""
    if train_data is not None and train_predictions is not None:
        # 訓練+テストデータの場合（2つのサブプロット）
        plt.figure(figsize=(15, 8))
        
        # 訓練データプロット
        plt.subplot(2, 1, 1)
        time_train = np.arange(len(train_data))
        plt.plot(time_train, train_data, 'b-', label='Ground Truth', alpha=0.7)
        plt.plot(time_train, train_predictions, 'r--', label='Prediction', alpha=0.7)
        plt.title('Training Data Prediction Results')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # テストデータプロット
        plt.subplot(2, 1, 2)
        time_test = time_steps + train_size
        plt.plot(time_test, ground_truth, 'b-', label='Ground Truth', alpha=0.7)
        plt.plot(time_test, predictions, 'r--', label='Prediction', alpha=0.7)
        plt.title('Test Data Prediction Results')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        # テストデータのみの場合（1つのプロット）
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, ground_truth, 'b-', label='Ground Truth', alpha=0.7)
        plt.plot(time_steps, predictions, 'r--', label='Prediction', alpha=0.7)
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('X-coordinate Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'outputs/{filename}', dpi=300, bbox_inches='tight')
    print(f"結果を 'outputs/{filename}' に保存しました。")
    plt.show()


def run_reservoir_demo(input_data, target_data, train_size, config, reg_param, 
                       title, filename, show_training=False):
    """
    Reservoir Computingのデモを実行する共通関数
    
    Args:
        input_data: 入力データ
        target_data: 目標データ
        train_size: 訓練データサイズ
        config: ReservoirConfigオブジェクト
        reg_param: 正則化パラメータ
        title: プロットのタイトル
        filename: 保存ファイル名
        show_training: 訓練データも表示するかどうか
    
    Returns:
        tuple: (train_mse, test_mse, train_mae, test_mae)
    """
    print(f"=== {title} ===")
    
    # データを正規化
    input_norm, _, _ = normalize_data(input_data)
    target_norm, target_mean, target_std = normalize_data(target_data)
    
    # 訓練データとテストデータに分割
    train_input = input_norm[:train_size]
    train_target = target_norm[:train_size]
    test_input = input_norm[train_size:]
    test_target = target_norm[train_size:]
    
    # Reservoir Computerを初期化
    rc = ReservoirComputer(config=config)
    
    print(f"Reservoir情報: {rc.get_reservoir_info()}")
    
    # 訓練
    print("訓練中...")
    rc.train(train_input, train_target, reg_param=reg_param)
    
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
    
    if show_training:
        train_predictions = rc.predict(train_input)
        train_predictions_orig = denormalize_data(train_predictions, target_mean, target_std)
        train_target_orig = denormalize_data(train_target, target_mean, target_std)
        
        train_mse = calculate_mse(train_predictions_orig, train_target_orig)
        train_mae = calculate_mae(train_predictions_orig, train_target_orig)
        
        print(f"訓練 MSE: {train_mse:.6f}, MAE: {train_mae:.6f}")
    
    print(f"テスト MSE: {test_mse:.6f}, MAE: {test_mae:.6f}")
    
    # 可視化
    if show_training:
        plot_prediction_results(
            test_target_orig, test_predictions, np.arange(len(test_target_orig)), 
            title, filename, train_target_orig, train_predictions_orig, train_size
        )
    else:
        plot_prediction_results(
            test_target_orig, test_predictions, np.arange(len(test_target_orig)), 
            title, filename
        )
    
    return train_mse, test_mse, train_mae, test_mae


def demo_sine_wave_prediction():
    """サイン波予測のデモンストレーション。"""
    # 設定ファイルから全パラメータを読み込み
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'sine_wave_demo_config.json')
    demo_config = DemoConfig.from_json(config_path)
    
    # データ生成（辞書として展開）
    data_params = demo_config.get_data_params()
    input_data, target_data = generate_sine_data(**data_params)
    
    # デモ実行
    return run_reservoir_demo(
        input_data, target_data, 
        train_size=demo_config.training.train_size,
        config=demo_config.reservoir,
        reg_param=demo_config.training.reg_param,
        title=demo_config.demo.title,
        filename=demo_config.demo.filename,
        show_training=demo_config.demo.show_training
    )


def demo_lorenz_prediction():
    """Lorenzアトラクターの予測デモンストレーション。"""
    # 設定ファイルから全パラメータを読み込み
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'lorenz_demo_config.json')
    demo_config = DemoConfig.from_json(config_path)
    
    # データ生成（辞書として展開）
    data_params = demo_config.get_data_params()
    input_data, target_data = generate_lorenz_data(**data_params)
    
    # X座標のみを使用
    input_data = input_data[:, 0:1]  # X座標のみ
    target_data = target_data[:, 0:1]  # X座標のみ
    
    # デモ実行
    return run_reservoir_demo(
        input_data, target_data,
        train_size=demo_config.training.train_size,
        config=demo_config.reservoir,
        reg_param=demo_config.training.reg_param,
        title=demo_config.demo.title,
        filename=demo_config.demo.filename,
        show_training=demo_config.demo.show_training
    )


def main():
    """メイン関数。"""
    print("JAXを使ったReservoir Computingのデモンストレーション")
    print("=" * 60)
    
    # GPU認識確認（JAX設定も含む）
    try:
        check_gpu_available()
        print("=" * 60)
    except RuntimeError as e:
        print(f"GPU認識エラー: {e}")
        print("GPU required for demo. Exiting...")
        sys.exit(1)
    
    # デモンストレーション実行
    try:
        # サイン波デモ
        sine_results = demo_sine_wave_prediction()
        
        # Lorenzデモ
        lorenz_results = demo_lorenz_prediction()
        
        print("\n" + "=" * 60)
        print("すべてのデモンストレーションが完了しました")
        print("=" * 60)
        
        # 結果サマリー
        print("結果サマリー:")
        if sine_results[0] is not None:  # 訓練結果がある場合
            print(f"サイン波 - 訓練 MSE: {sine_results[0]:.6f}, テスト MSE: {sine_results[1]:.6f}")
        else:
            print(f"サイン波 - テスト MSE: {sine_results[1]:.6f}")
        print(f"Lorenz - テスト MSE: {lorenz_results[1]:.6f}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
