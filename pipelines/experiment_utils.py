"""Shared experiment helpers for runners and plotting."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from core_lib.core import ExperimentConfig
from pipelines.plotting import plot_classification_results


def _resolve_model_kind(model_name: str, model_type: str) -> str:
    """Resolve a coarse model kind for regression pipeline dispatch."""
    mt = (model_type or "").lower()
    name = (model_name or "").lower()
    if "analog" in name or "analog" in mt:
        return "analog_quantum_legacy"
    if "quantum" in mt:
        return "gatebased_quantum"
    return "classical"


def _log_ridge_search(model: Any) -> Optional[list]:
    """Print ridge-search diagnostics if available and return the log."""
    ridge_log = getattr(model, "ridge_search_log", None)
    if ridge_log:
        print("Ridge λ grid search")
        for entry in ridge_log:
            lam = entry["lambda"]
            if "val_accuracy" in entry:
                print(f"  λ={lam:.2e} -> val Acc={entry['val_accuracy']:.4f}")
            elif "val_mse" in entry:
                print(f"  λ={lam:.2e} -> val MSE={entry['val_mse']:.6f}")
            elif "train_accuracy" in entry:
                print(f"  λ={lam:.2e} -> train Acc={entry['train_accuracy']:.4f}")
            elif "train_mse" in entry:
                print(f"  λ={lam:.2e} -> train MSE={entry['train_mse']:.6f}")
            else:
                print(f"  λ={lam:.2e}")
        best_lambda = getattr(model, "best_ridge_lambda", None)
        if best_lambda is not None:
            print(f"Selected λ={best_lambda:.2e}")
    return ridge_log


def _json_default(obj: Any):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _save_config_snapshot(
    config: ExperimentConfig,
    output_filename: str,
    metrics: Dict[str, Any],
    ridge_log: Optional[list],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    snapshot = config.model_dump(mode='json')
    snapshot.setdefault('results', {})
    snapshot['results']['metrics'] = metrics
    if ridge_log is not None:
        snapshot['results']['ridge_search'] = ridge_log
    if extra:
        snapshot['results'].update(extra)

    snapshot_path = Path('outputs') / f"{Path(output_filename).stem}_config.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open('w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, default=_json_default)
    print(f"Saved config snapshot -> {snapshot_path}")


def _helper_plot_classification(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_pred: np.ndarray,
    test_pred: np.ndarray,
    title: str,
    filename: str,
    metrics_dict: Dict[str, float],
    val_labels: Optional[np.ndarray] = None,
    val_pred: Optional[np.ndarray] = None,
    ridge_lambda: Optional[float] = None,
) -> None:
    """Shared helper to plot confusion matrix and metrics for classification runs."""
    labels_arrays = [
        np.asarray(train_labels),
        np.asarray(test_labels),
        np.asarray(train_pred),
        np.asarray(test_pred),
    ]
    if val_labels is not None:
        labels_arrays.append(np.asarray(val_labels))
    if val_pred is not None:
        labels_arrays.append(np.asarray(val_pred))

    detected_classes = max(
        (int(arr.max()) if arr.size > 0 else -1) for arr in labels_arrays
    )
    class_count = max(detected_classes + 1, 10)
    class_names = [str(i) for i in range(class_count)]

    metrics_info: Dict[str, Any] = {
        "Train MSE": float(metrics_dict["train_mse"]),
        "Test MSE": float(metrics_dict["test_mse"]),
        "Train Acc": f"{float(metrics_dict['train_accuracy']):.4f}",
        "Test Acc": f"{float(metrics_dict['test_accuracy']):.4f}",
    }
    if "val_mse" in metrics_dict:
        metrics_info["Val MSE"] = float(metrics_dict["val_mse"])
    if "val_accuracy" in metrics_dict:
        metrics_info["Val Acc"] = f"{float(metrics_dict['val_accuracy']):.4f}"
    if ridge_lambda is not None:
        metrics_info["Ridge λ"] = f"{ridge_lambda:.2e}"

    plot_classification_results(
        np.asarray(train_labels),
        np.asarray(test_labels),
        np.asarray(train_pred),
        np.asarray(test_pred),
        title,
        filename,
        metrics_info=metrics_info,
        class_names=class_names,
    )
