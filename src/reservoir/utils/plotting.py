"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/utils/plotting.py
Classification visualization utilities with validation support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _find_project_root() -> Path:
    """Locate the project root (directory containing pyproject.toml)."""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path(__file__).parent.parent


PROJECT_ROOT = _find_project_root()


def _resolve_output_path(filename: str) -> Path:
    """Resolve output path under the project root."""
    path = Path(filename)
    if path.is_absolute() or (len(path.parts) > 0 and path.parts[0] == "outputs"):
        return PROJECT_ROOT / path
    return PROJECT_ROOT / "outputs" / path


def plot_classification_results(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_predictions: np.ndarray,
    test_predictions: np.ndarray,
    title: str,
    filename: str,
    metrics_info: Optional[Dict[str, Any]] = None,
    class_names: Optional[Sequence[str]] = None,
    val_labels: Optional[np.ndarray] = None,
    val_predictions: Optional[np.ndarray] = None,
    best_lambda: Optional[float] = None,
    lambda_norm: Optional[float] = None,
) -> None:
    """
    Visualize classification results with confusion matrix and accuracy bars, with optional validation.
    Annotates the selected ridge lambda and weight norm when provided.
    """

    def _to_numpy(array: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
        if dtype is None:
            return np.asarray(array)
        return np.asarray(array, dtype=dtype)

    train_labels_np = _to_numpy(train_labels, dtype=np.int32)
    test_labels_np = _to_numpy(test_labels, dtype=np.int32)
    train_predictions_np = _to_numpy(train_predictions, dtype=np.int32)
    test_predictions_np = _to_numpy(test_predictions, dtype=np.int32)

    val_labels_np = _to_numpy(val_labels, dtype=np.int32) if val_labels is not None else None
    val_predictions_np = _to_numpy(val_predictions, dtype=np.int32) if val_predictions is not None else None

    def _safe_max(array: np.ndarray) -> int:
        return int(array.max()) if array.size > 0 else -1

    if class_names is None:
        inferred_max = max(
            _safe_max(train_labels_np),
            _safe_max(test_labels_np),
            _safe_max(train_predictions_np),
            _safe_max(test_predictions_np),
            _safe_max(val_labels_np) if val_labels_np is not None else -1,
            _safe_max(val_predictions_np) if val_predictions_np is not None else -1,
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

    def _calc_acc(true_y, pred_y):
        if true_y.size == 0:
            return 0.0
        return float(np.mean(true_y == pred_y))

    train_accuracy = _calc_acc(train_labels_np, train_predictions_np)
    test_accuracy = _calc_acc(test_labels_np, test_predictions_np)
    val_accuracy = None
    if val_labels_np is not None and val_predictions_np is not None:
        val_accuracy = _calc_acc(val_labels_np, val_predictions_np)
    elif metrics_info and "Val Acc" in metrics_info:
        try:
            val_accuracy = float(metrics_info["Val Acc"])
            if val_accuracy > 1.0:
                val_accuracy = val_accuracy / 100.0
        except (TypeError, ValueError):
            val_accuracy = None

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

    threshold = (confusion_matrix.max() / 2.0) if confusion_matrix.max() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            value = confusion_matrix[i, j]
            color = 'white' if value > threshold else 'black'
            ax_conf.text(j, i, str(value), ha='center', va='center', color=color, fontsize=9)

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

    summary_parts = []
    if best_lambda is not None:
        lambda_str = f"lambda*: {best_lambda:.3e}"
        norm_str = f" ||w||={float(lambda_norm):.3e}" if lambda_norm is not None else ""
        summary_parts.append(lambda_str + norm_str)
    if summary_parts:
        fig.text(0.5, 0.92, " | ".join(summary_parts), ha='center', fontsize=10, color='gray')

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))

    output_path = _resolve_output_path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"分類結果を '{output_path}' に保存しました。")


def plot_loss_history(history: Sequence[float], filename: str, title: str = "Loss Curve", learning_rate: Optional[float] = None) -> None:
    """Plot a generic loss history curve and save to outputs."""
    output_path = _resolve_output_path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.yscale("log")
    
    # Add learning rate to title if provided
    full_title = title
    if learning_rate is not None:
        full_title = f"{title} (LR={learning_rate:.4f})"
    plt.title(full_title)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved to '{output_path}'.")


def plot_timeseries_comparison(
    targets: np.ndarray,
    predictions: np.ndarray,
    filename: str,
    title: str = "TimeSeries Prediction",
    max_features: int = 3,
    start_step: int = 0,
    end_step: Optional[int] = None,
    time_offset: int = 0,
) -> None:
    """
    Plot predicted vs actual time series trajectories.
    Handles 3D (Batch, Time, Feat) or 2D (Time, Feat) inputs.
    Plots the first sample (if batched) and up to `max_features` features.
    """
    targets_np = np.asarray(targets)
    preds_np = np.asarray(predictions)

    # Standardize to (Time, Feat)
    if targets_np.ndim == 3:
        # Take first sample in batch
        target_seq = targets_np[0]
        pred_seq = preds_np[0]
    elif targets_np.ndim == 2:
        target_seq = targets_np
        pred_seq = preds_np
    else:
        print(f"Skipping plot: expected 2D/3D arrays, got {targets_np.shape}")
        return

    time_steps, n_feats = target_seq.shape
    plot_feats = min(n_feats, max_features)
    
    if end_step is None:
        end_step = time_steps

    # Slice time (0-based for array indexing)
    target_slice = target_seq[start_step:end_step]
    pred_slice = pred_seq[start_step:end_step]
    
    # Time axis (shifted by global offset)
    t_start = start_step + time_offset
    t_end = t_start + (end_step - start_step)
    t_axis = np.arange(t_start, t_end)

    fig, axes = plt.subplots(plot_feats, 1, figsize=(12, 3 * plot_feats), sharex=True)
    if plot_feats == 1:
        axes = [axes]

    for i in range(plot_feats):
        ax = axes[i]
        ax.plot(t_axis, target_slice[:, i], label="Actual", color="black", alpha=0.7)
        ax.plot(t_axis, pred_slice[:, i], label="Predicted", color="tab:blue", alpha=0.9, linestyle="--")
        ax.set_ylabel(f"Feature {i}")
        ax.grid(True, linestyle=":", alpha=0.5)
        if i == 0:
            ax.legend(loc="upper right")
        if i == plot_feats - 1:
            ax.set_xlabel("Time Step")

    output_path = _resolve_output_path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Prediction plot saved to '{output_path}'.")
