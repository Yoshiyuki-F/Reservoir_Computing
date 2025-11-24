"""
Reservoir Computing用の可視化ユーティリティ。
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Any, Sequence
from pathlib import Path


def _find_project_root() -> Path:
    """Locate the project root (directory containing pyproject.toml)."""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path(__file__).parent.parent


PROJECT_ROOT = _find_project_root()


def _resolve_output_path(filename: str) -> Path:
    """統一された出力パス解決ロジック"""
    path = Path(filename)
    # 既に絶対パスか、outputsで始まっている場合はそのまま
    if path.is_absolute() or (len(path.parts) > 0 and path.parts[0] == "outputs"):
        return PROJECT_ROOT / path
    # それ以外は outputs/ 下に保存
    return PROJECT_ROOT / "outputs" / path


def plot_epoch_metric(
    epochs: Sequence[int],
    values: Sequence[float],
    title: str,
    filename: str,
    *,
    ylabel: str = "Metric",
    metric_name: str = "value",
    extra_metrics: Optional[Dict[str, Sequence[float]]] = None,
    phase2_test_acc: Optional[float] = None,
    phase2_train_acc: Optional[float] = None,
) -> None:
    """Plot a simple training curve (e.g., epoch vs accuracy) and save as PNG."""
    if not epochs or not values:
        return

    epochs_arr = np.asarray(list(epochs), dtype=np.int32)
    values_arr = np.asarray(list(values), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        epochs_arr,
        values_arr,
        marker="o",
        linestyle="-",
        color="tab:blue",
        label=metric_name,
    )

    if extra_metrics:
        colors = [
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
        ]
        for idx, (name, series) in enumerate(extra_metrics.items()):
            vals = np.asarray(list(series), dtype=np.float64)
            if vals.size == 0:
                continue
            length = min(len(epochs_arr), len(vals))
            color = colors[idx % len(colors)]
            ax.plot(
                epochs_arr[:length],
                vals[:length],
                marker="o",
                linestyle="--",
                color=color,
                label=name,
            )
    x_ticks = list(epochs_arr)

    # Phase 2 マーカーの表示
    if phase2_test_acc is not None or phase2_train_acc is not None:
        x_phase2 = int(epochs_arr[-1]) + 1
        if phase2_test_acc is not None:
            ax.scatter(
                [x_phase2],
                [float(phase2_test_acc)],
                color="tab:blue",
                marker="*",
                s=80,
                label="phase2_test_acc",
                zorder=5,
            )
        if phase2_train_acc is not None:
            ax.scatter(
                [x_phase2],
                [float(phase2_train_acc)],
                color="tab:orange",
                marker="*",
                s=80,
                label="phase2_train_acc",
                zorder=5,
            )
        x_ticks.append(x_phase2)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    # 目盛りが多すぎる場合は間引く
    if len(x_ticks) > 20:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        ax.set_xticks(x_ticks)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    output_path = _resolve_output_path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"学習曲線を '{output_path}' に保存しました。")


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
    """Reservoir Computingの予測結果を可視化して保存。"""

    def _to_numpy(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if array is None:
            return None
        np_array = np.asarray(array, dtype=np.float64)
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
            2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.25},
        )
    else:
        fig, ax_main = plt.subplots(figsize=(12, 6))
        ax_zoom = None

    if train_data is not None and train_predictions is not None:
        train_steps = len(train_data)
        offset = train_size if train_size is not None else train_steps
        boundary_x = offset - 0.5 if train_size is not None else 0.0
        time_train = np.arange(train_steps)
        time_test = np.arange(len(ground_truth)) + offset

        ax_main.plot(time_train, train_data, label='Ground Truth', color='tab:blue', alpha=0.85)
        ax_main.plot(time_test, ground_truth, color='tab:blue', alpha=0.85)
        ax_main.plot(time_train, train_predictions, label='Train Prediction', color='tab:orange', linestyle='--', alpha=0.9)
        ax_main.plot(time_test, predictions, label='Test Prediction', color='tab:red', linestyle='--', alpha=0.85)

        if train_size is not None:
            ax_main.axvline(boundary_x, color='k', linestyle=':', alpha=0.7, label='Train/Test Split')

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
            ax_zoom.plot(zoom_time[mask], zoom_pred[mask], color='tab:red', linestyle='--', alpha=0.8)
            ax_zoom.set_title('Zoomed Test Region', fontsize=11)
            ax_zoom.set_xlabel('Sequence Steps')
            ax_zoom.set_ylabel(y_axis_label)
            ax_zoom.grid(True, alpha=0.3)
    else:
        time_offset = train_size if train_size is not None else 0
        boundary_x = (train_size - 0.5) if train_size is not None else 0.0
        time_axis = time_steps + time_offset
        ax_main.plot(time_axis, ground_truth, 'b-', label='Ground Truth', alpha=0.85)
        ax_main.plot(time_axis, predictions, 'r--', label='Test Prediction', alpha=0.8)
        if train_size is not None:
            ax_main.axvline(boundary_x, color='k', linestyle=':', alpha=0.7, label='Train/Test Split')

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
    output_path = _resolve_output_path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"結果を '{output_path}' に保存しました。")


def plot_classification_results(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_predictions: np.ndarray,
    test_predictions: np.ndarray,
    title: str,
    filename: str,
    metrics_info: Optional[Dict[str, Any]] = None,
    class_names: Optional[Sequence[str]] = None,
    # ▼▼▼ 追加された引数 ▼▼▼
    val_labels: Optional[np.ndarray] = None,
    val_predictions: Optional[np.ndarray] = None,
) -> None:
    """可視化: 分類タスク用の混同行列と精度バーを描画して保存。"""

    def _to_numpy(array: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
        if dtype is None:
            return np.asarray(array)
        return np.asarray(array, dtype=dtype)

    train_labels_np = _to_numpy(train_labels, dtype=np.int32)
    test_labels_np = _to_numpy(test_labels, dtype=np.int32)
    train_predictions_np = _to_numpy(train_predictions, dtype=np.int32)
    test_predictions_np = _to_numpy(test_predictions, dtype=np.int32)

    # クラス数の自動判定
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

    # 混同行列 (Testデータのみ)
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

    # 精度の計算
    def _calc_acc(true_y, pred_y):
        if true_y.size == 0: return 0.0
        return float(np.mean(true_y == pred_y))

    train_accuracy = _calc_acc(train_labels_np, train_predictions_np)
    test_accuracy = _calc_acc(test_labels_np, test_predictions_np)

    val_accuracy = None
    if val_labels is not None and val_predictions is not None:
        val_labels_np = _to_numpy(val_labels, dtype=np.int32)
        val_pred_np = _to_numpy(val_predictions, dtype=np.int32)
        val_accuracy = _calc_acc(val_labels_np, val_pred_np)
    elif metrics_info and "Val Acc" in metrics_info:
        try:
            val_accuracy = float(metrics_info["Val Acc"])
            if val_accuracy > 1.0:
                val_accuracy = val_accuracy / 100.0
        except (TypeError, ValueError):
            val_accuracy = None

    # Plotting
    fig, (ax_conf, ax_bar) = plt.subplots(1, 2, figsize=(12, 6))

    # 1. Confusion Matrix (Test)
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

    threshold = (confusion_matrix.max() / 2.0) if confusion_matrix.max() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            value = confusion_matrix[i, j]
            color = 'white' if value > threshold else 'black'
            ax_conf.text(j, i, str(value), ha='center', va='center', color=color, fontsize=9)

    # 2. Accuracy Bar Chart
    accuracy_labels = ['Train', 'Test']
    accuracy_values = [train_accuracy * 100.0, test_accuracy * 100.0]
    bar_colors = ['tab:green', 'tab:orange']

    if val_accuracy is not None:
        accuracy_labels.insert(1, 'Val')
        accuracy_values.insert(1, val_accuracy * 100.0)
        bar_colors.insert(1, 'tab:purple')

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

    # メトリクス情報の表示 (Validation Accuracyが渡されていない場合のみ metrics_info から補完表示する等のロジックを入れても良いが、
    # 今回はシンプルに下部にテキスト表示する)
    if metrics_info:
        info_text = " | ".join([f"{k}: {v}" for k, v in metrics_info.items()])
        # タイトルの下に小さく表示
        fig.text(0.5, 0.92, info_text, ha='center', fontsize=10, color='gray')

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))

    output_path = _resolve_output_path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"分類結果を '{output_path}' に保存しました。")
