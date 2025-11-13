"""
Reservoir Computing用の可視化ユーティリティ。
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Any, Sequence, Tuple
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
    train_size: Optional[int] = None,
    y_axis_label: str = "Value",
    metrics_info: Optional[Dict[str, Any]] = None,
    add_test_zoom: bool = False,
    zoom_range: Optional[Sequence[int]] = None,
) -> None:
    """Reservoir Computingの予測結果を可視化して保存。
    
    訓練データとテストデータの予測結果を一続きの時間軸で表示し、
    高解像度画像として保存します。訓練データが提供されない場合は、
    テストデータのみを単独でプロットします。
    
    Args:
        ground_truth: 実際の目標値の配列
        predictions: モデルの予測値の配列
        time_steps: 時刻ステップの配列
        title: プロットのメインタイトル
        filename: 保存する画像ファイル名（拡張子を含む）
        train_data: 訓練データの実際の値（オプション）
        train_predictions: 訓練データの予測値（オプション）
        train_size: 訓練データのサイズ（オプション、テスト時刻軸調整用）
        y_axis_label: 縦軸に表示するラベル
        
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
    def _to_numpy(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Convert to numpy array and squeeze trailing singleton dims."""
        if array is None:
            return None

        np_array = np.asarray(array, dtype=np.float64)

        # 単一特徴量の場合は余分な次元を削除
        if np_array.ndim > 1 and np_array.shape[-1] == 1:
            np_array = np_array.reshape(np_array.shape[0], -1)
            if np_array.shape[1] == 1:
                np_array = np_array.squeeze(-1)

        return np_array

    ground_truth = _to_numpy(ground_truth)
    predictions = _to_numpy(predictions)
    train_data = _to_numpy(train_data)
    train_predictions = _to_numpy(train_predictions)

    # Prepare figure/axes
    if add_test_zoom and train_size is not None:
        fig, (ax_main, ax_zoom) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.25},
        )
    else:
        fig, ax_main = plt.subplots(figsize=(12, 6))
        ax_zoom = None

    if train_data is not None and train_predictions is not None:
        train_steps = len(train_data)
        offset = train_size if train_size is not None else train_steps
        time_train = np.arange(train_steps)
        time_test = np.arange(len(ground_truth)) + offset

        ax_main.plot(
            time_train,
            train_data,
            label='Ground Truth',
            color='tab:blue',
            alpha=0.85,
        )
        ax_main.plot(
            time_test,
            ground_truth,
            color='tab:blue',
            alpha=0.85,
        )

        ax_main.plot(
            time_train,
            train_predictions,
            label='Train Prediction',
            color='tab:orange',
            linestyle='--',
            alpha=0.9,
        )
        ax_main.plot(
            time_test,
            predictions,
            label='Test Prediction',
            color='tab:red',
            linestyle='--',
            alpha=0.85,
        )

        if train_size is not None:
            boundary_x = offset - 0.5
            ax_main.axvline(
                boundary_x,
                color='k',
                linestyle=':',
                alpha=0.7,
                label='Train/Test Split',
            )

        if ax_zoom is not None:
            zoom_time = time_test
            zoom_truth = ground_truth
            zoom_pred = predictions

            if zoom_range is not None and len(zoom_range) == 2:
                start, end = zoom_range
            else:
                start = max(int(zoom_time[-1] - 200), int(boundary_x)) if len(zoom_time) > 0 else 0
                end = int(zoom_time[-1]) if len(zoom_time) > 0 else start + 1

            mask = (zoom_time >= start) & (zoom_time <= end)
            if not np.any(mask):
                mask = np.ones_like(zoom_time, dtype=bool)

            ax_zoom.plot(zoom_time[mask], zoom_truth[mask], color='tab:blue', alpha=0.85)
            ax_zoom.plot(
                zoom_time[mask],
                zoom_pred[mask],
                color='tab:red',
                linestyle='--',
                alpha=0.8,
            )
            ax_zoom.set_title('Zoomed Test Region', fontsize=11)
            ax_zoom.set_xlabel('Sequence Steps')
            ax_zoom.set_ylabel(y_axis_label)
            ax_zoom.grid(True, alpha=0.3)
    else:
        time_offset = train_size if train_size is not None else 0
        time_axis = time_steps + time_offset
        ax_main.plot(time_axis, ground_truth, 'b-', label='Ground Truth', alpha=0.85)
        ax_main.plot(time_axis, predictions, 'r--', label='Test Prediction', alpha=0.8)

        if train_size is not None:
            boundary_x = train_size - 0.5
            ax_main.axvline(
                boundary_x,
                color='k',
                linestyle=':',
                alpha=0.7,
                label='Train/Test Split',
            )

        if ax_zoom is not None:
            start = max(int(time_axis[-1] - 200), int(boundary_x)) if len(time_axis) > 0 else 0
            end = int(time_axis[-1]) if len(time_axis) > 0 else start + 1
            mask = (time_axis >= start) & (time_axis <= end)
            if not np.any(mask):
                mask = np.ones_like(time_axis, dtype=bool)
            ax_zoom.plot(time_axis[mask], ground_truth[mask], color='tab:blue', alpha=0.85)
            ax_zoom.plot(time_axis[mask], predictions[mask], 'r--', alpha=0.8)
            ax_zoom.set_title('Zoomed Test Region', fontsize=11)
            ax_zoom.set_xlabel('Sequence Steps')
            ax_zoom.set_ylabel(y_axis_label)
            ax_zoom.grid(True, alpha=0.3)

    ax_main.set_title(title)
    ax_main.set_xlabel('Sequence Steps')
    ax_main.set_ylabel(y_axis_label)
    ax_main.legend(loc='upper left')
    ax_main.grid(True, alpha=0.3)

    if metrics_info:
        caption_lines = []
        for key, value in metrics_info.items():
            if isinstance(value, float):
                caption_lines.append(f"{key}: {value:.6f}")
            else:
                caption_lines.append(f"{key}: {value}")
        caption_text = "\n".join(caption_lines)
        fig.text(0.02, 0.02 if ax_zoom is not None else 0.05, caption_text, fontsize=10, family='monospace')

    fig.tight_layout()
    output_path = PROJECT_ROOT / f'outputs/{filename}'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"結果を 'outputs/{filename}' に保存しました。")


def plot_classification_results(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_predictions: np.ndarray,
    test_predictions: np.ndarray,
    title: str,
    filename: str,
    metrics_info: Optional[Dict[str, Any]] = None,
    class_names: Optional[Sequence[str]] = None,
) -> None:
    """可視化: 分類タスク用の混同行列と精度バーを描画して保存。"""

    def _to_numpy(array: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """Convert JAX/NumPy arraylikes to numpy ndarray."""
        if dtype is None:
            return np.asarray(array)
        return np.asarray(array, dtype=dtype)

    train_labels_np = _to_numpy(train_labels, dtype=np.int32)
    test_labels_np = _to_numpy(test_labels, dtype=np.int32)
    train_predictions_np = _to_numpy(train_predictions, dtype=np.int32)
    test_predictions_np = _to_numpy(test_predictions, dtype=np.int32)

    def _safe_max(array: np.ndarray) -> int:
        return int(array.max()) if array.size > 0 else -1

    if class_names is None:
        inferred_max = max(
            _safe_max(train_labels_np),
            _safe_max(test_labels_np),
            _safe_max(train_predictions_np),
            _safe_max(test_predictions_np),
        )
        num_classes = max(inferred_max + 1, 1)
        class_names = [str(idx) for idx in range(num_classes)]
    else:
        num_classes = len(class_names)

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    valid_mask = (
        (test_labels_np >= 0)
        & (test_labels_np < num_classes)
        & (test_predictions_np >= 0)
        & (test_predictions_np < num_classes)
    )
    np.add.at(
        confusion_matrix,
        (test_labels_np[valid_mask], test_predictions_np[valid_mask]),
        1,
    )

    train_accuracy = (
        float(np.mean(train_predictions_np == train_labels_np))
        if train_labels_np.size > 0
        else 0.0
    )
    test_accuracy = (
        float(np.mean(test_predictions_np == test_labels_np))
        if test_labels_np.size > 0
        else 0.0
    )

    fig, (ax_conf, ax_bar) = plt.subplots(1, 2, figsize=(12, 6))

    im = ax_conf.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    cbar = fig.colorbar(im, ax=ax_conf, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Count', rotation=-90, va='bottom')

    ax_conf.set_title('Test Confusion Matrix')
    ax_conf.set_xlabel('Predicted Label')
    ax_conf.set_ylabel('True Label')
    ax_conf.set_xticks(range(num_classes))
    ax_conf.set_yticks(range(num_classes))
    ax_conf.set_xticklabels(class_names, rotation=45, ha='right')
    ax_conf.set_yticklabels(class_names)

    if confusion_matrix.max() > 0:
        threshold = confusion_matrix.max() / 2.0
    else:
        threshold = 0.5

    for i in range(num_classes):
        for j in range(num_classes):
            value = confusion_matrix[i, j]
            color = 'white' if value > threshold else 'black'
            ax_conf.text(
                j,
                i,
                str(value),
                ha='center',
                va='center',
                color=color,
                fontsize=9,
            )

    accuracy_labels = ['Train', 'Test']
    accuracy_values = [train_accuracy * 100.0, test_accuracy * 100.0]
    bar_colors = ['tab:green', 'tab:orange']

    val_accuracy = None
    if metrics_info:
        val_entry = metrics_info.get("Val Acc")
        if val_entry is not None:
            try:
                val_accuracy = float(val_entry)
            except (ValueError, TypeError):
                val_accuracy = None
    if val_accuracy is not None:
        accuracy_labels.append('Val')
        accuracy_values.append(
            val_accuracy * 100.0 if val_accuracy <= 1.0 else val_accuracy
        )
        bar_colors.append('tab:purple')

    accuracy_values = np.array(accuracy_values)
    bars = ax_bar.barh(accuracy_labels, accuracy_values, color=bar_colors)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel('Accuracy (%)')
    ax_bar.set_title('Accuracy Overview')
    ax_bar.grid(True, axis='x', alpha=0.3, linestyle='--')

    for bar, accuracy in zip(bars, accuracy_values):
        display_value = min(accuracy + 1.0, 99.9)
        ax_bar.text(
            display_value,
            bar.get_y() + bar.get_height() / 2,
            f"{accuracy:.2f}%",
            va='center',
            fontsize=10,
        )

    fig.suptitle(title, fontsize=16)

    # Metrics caption removed per design request

    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    output_path = PROJECT_ROOT / f'outputs/{filename}'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"分類結果を 'outputs/{filename}' に保存しました。")
