"""
Reservoir Computing用の可視化ユーティリティ。
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def plot_prediction_results(
    ground_truth: np.ndarray, 
    predictions: np.ndarray, 
    time_steps: np.ndarray, 
    title: str, 
    filename: str,
    train_data: Optional[np.ndarray] = None, 
    train_predictions: Optional[np.ndarray] = None, 
    train_size: Optional[int] = None
) -> None:
    """Reservoir Computingの予測結果を可視化して保存。
    
    訓練データとテストデータの予測結果を比較プロットとして表示し、
    高解像度画像として保存します。訓練データが提供された場合は
    2段構成、テストデータのみの場合は1段構成でプロットします。
    
    Args:
        ground_truth: 実際の目標値の配列
        predictions: モデルの予測値の配列
        time_steps: 時刻ステップの配列
        title: プロットのメインタイトル
        filename: 保存する画像ファイル名（拡張子を含む）
        train_data: 訓練データの実際の値（オプション）
        train_predictions: 訓練データの予測値（オプション）
        train_size: 訓練データのサイズ（オプション、テスト時刻軸調整用）
        
    Examples:
        テストデータのみをプロット:
        
        >>> import numpy as np
        >>> ground_truth = np.sin(np.linspace(0, 10, 100))
        >>> predictions = ground_truth + 0.1 * np.random.randn(100)
        >>> time_steps = np.arange(100)
        >>> plot_prediction_results(
        ...     ground_truth, predictions, time_steps,
        ...     "Sin Wave Prediction", "test_results.png"
        ... )
        
        訓練+テストデータをプロット:
        
        >>> plot_prediction_results(
        ...     test_ground_truth, test_predictions, test_time,
        ...     "Complete Results", "full_results.png",
        ...     train_data=train_ground_truth,
        ...     train_predictions=train_pred,
        ...     train_size=800
        ... )
        
    Note:
        - 画像はoutputs/ディレクトリに300dpiで保存されます
        - outputs/ディレクトリが存在しない場合は事前に作成してください
        - プロット完了後にmatplotlibのshow()が呼ばれます
    """
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
        time_test = time_steps + train_size if train_size is not None else time_steps
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
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT/ f'outputs/{filename}', dpi=300, bbox_inches='tight')
    print(f"結果を 'outputs/{filename}' に保存しました。")