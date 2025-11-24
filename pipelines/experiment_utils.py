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
        return "gate_based_quantum"
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

    output_path = Path(output_filename)
    snapshot_path = output_path.with_name(f"{output_path.stem}_config.json")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open('w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, default=_json_default)
    print(f"Saved config snapshot -> {snapshot_path}")


def _save_json_snapshot(snapshot_path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON snapshot to disk with project defaults."""
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"Saved config snapshot -> {snapshot_path}")


def _plot_classification_confusion(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_pred: np.ndarray,
    test_pred: np.ndarray,
    *,
    title: str,
    filename: str | Path,
    metrics_dict: Dict[str, float],
    val_labels: Optional[np.ndarray] = None,
    val_pred: Optional[np.ndarray] = None,
    ridge_lambda: Optional[float] = None,
) -> None:
    """Render and save classification confusion and metrics visualization."""
    _helper_plot_classification(
        train_labels,
        test_labels,
        train_pred,
        test_pred,
        title,
        str(filename),
        metrics_dict,
        val_labels=val_labels,
        val_pred=val_pred,
        ridge_lambda=ridge_lambda,
    )


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
    val_accuracy_value = metrics_dict.get("val_accuracy")
    if val_accuracy_value is None and val_labels is not None and val_pred is not None:
        try:
            val_accuracy_value = float(np.mean(np.asarray(val_pred) == np.asarray(val_labels)))
        except Exception:
            val_accuracy_value = None
    if val_accuracy_value is not None:
        metrics_info["Val Acc"] = float(val_accuracy_value)
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


def finalize_classification_report(
    output_filename: str | Path,
    plot_title: str,
    metrics: Dict[str, Any],
    *,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_pred: np.ndarray,
    test_pred: np.ndarray,
    val_labels: Optional[np.ndarray] = None,
    val_pred: Optional[np.ndarray] = None,
    ridge_lambda: Optional[float] = None,
    ridge_log: Optional[list] = None,
    config: Optional[ExperimentConfig] = None,
    snapshot_payload: Optional[Dict[str, Any]] = None,
    extra_results: Optional[Dict[str, Any]] = None,
) -> None:
    """Unified classification reporting: confusion plot + snapshot write."""
    if config is not None and snapshot_payload is not None:
        raise ValueError("Provide either 'config' or 'snapshot_payload', not both.")

    base_path = Path(output_filename)
    suffix = base_path.suffix or ".png"
    base_path = base_path if base_path.suffix else base_path.with_suffix(suffix)
    confusion_filename = base_path.with_name(f"{base_path.stem}_confusion{suffix}")

    _plot_classification_confusion(
        np.asarray(train_labels),
        np.asarray(test_labels),
        np.asarray(train_pred),
        np.asarray(test_pred),
        title=plot_title,
        filename=confusion_filename,
        metrics_dict=metrics,
        val_labels=np.asarray(val_labels) if val_labels is not None else None,
        val_pred=np.asarray(val_pred) if val_pred is not None else None,
        ridge_lambda=ridge_lambda,
    )

    metrics_copy = dict(metrics)
    if config is not None:
        _save_config_snapshot(
            config,
            str(base_path),
            metrics_copy,
            ridge_log,
            extra=extra_results,
        )
        return

    payload = dict(snapshot_payload or {})
    results_section = dict(payload.get("results", {}))
    results_section["metrics"] = metrics_copy
    if ridge_log is not None:
        results_section["ridge_search"] = ridge_log
    if extra_results:
        results_section.update(extra_results)
    payload["results"] = results_section

    snapshot_path = base_path.with_name(f"{base_path.stem}_config.json")
    _save_json_snapshot(snapshot_path, payload)
