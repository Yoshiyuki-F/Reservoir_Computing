"""
Reporting utilities for post-run analysis: plotting, logging, and file outputs.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np


def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def print_ridge_search_results(train_res: Dict[str, Any], metric: str) -> None:
    if not isinstance(train_res, dict):
        return
    history = train_res.get("search_history")
    if not history:
        return
    best_lam = train_res.get("best_lambda")
    weight_norms = train_res.get("weight_norms", {}) or {}
    metric_label = "Accuracy" if metric == "accuracy" else "MSE"
    best_by_metric = None
    if metric == "accuracy":
        best_by_metric = max(history, key=history.get)
    else:
        best_by_metric = min(history, key=history.get)

    best_marker = best_lam if best_lam is not None else best_by_metric

    print("\n" + "=" * 40)
    print(f"Ridge Hyperparameter Search ({metric_label})")
    print("-" * 40)
    sorted_lambdas = sorted(history.keys())
    for lam in sorted_lambdas:
        score = float(history[lam])
        norm = weight_norms.get(lam)
        norm_str = f"(Norm: {norm:.2e})" if norm is not None else "(Norm: n/a)"
        marker = " <= best" if (best_marker is not None and abs(float(lam) - float(best_marker)) < 1e-12) else ""
        print(f"   λ = {float(lam):.2e} : Val Score = {score:.4f} {norm_str}{marker}")
    print("=" * 40 + "\n")


def plot_distillation_loss(training_logs: Dict[str, Any], save_path: str, title: str) -> None:
    if not isinstance(training_logs, dict):
        return
    loss_history = training_logs.get("loss_history")
    if not loss_history:
        return
    try:
        from reservoir.utils.plotting import plot_loss_history
    except Exception as exc:  # pragma: no cover
        print(f"Skipping distillation loss plotting due to import error: {exc}")
        return
    plot_loss_history(loss_history, save_path, title=title)


def plot_classification_report(
    *,
    runner: Any,
    readout: Any,
    train_X: Any,
    train_y: Any,
    test_X: Any,
    test_y: Any,
    val_X: Optional[Any],
    val_y: Optional[Any],
    filename: str,
    model_type_str: str,
    dataset_name: str,
    metric: str,
    selected_lambda: Optional[float],
    lambda_norm: Optional[float],
    results: Dict[str, Any],
    training_obj: Any,
) -> None:
    try:
        from reservoir.utils.plotting import plot_classification_results
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plotting due to import error: {exc}")
        return

    feature_batch_size = int(getattr(training_obj, "batch_size", 0) or 0)

    train_labels_np = np.asarray(train_y)
    test_labels_np = np.asarray(test_y)
    if train_labels_np.ndim > 1:
        train_labels_np = np.argmax(train_labels_np, axis=-1)
    if test_labels_np.ndim > 1:
        test_labels_np = np.argmax(test_labels_np, axis=-1)

    train_features_np = runner.batch_transform(train_X, batch_size=feature_batch_size)
    test_features_np = runner.batch_transform(test_X, batch_size=feature_batch_size)
    if readout is None:
        train_pred_np = np.asarray(train_features_np)
        test_pred_np = np.asarray(test_features_np)
    else:
        train_pred_np = np.asarray(readout.predict(train_features_np))
        test_pred_np = np.asarray(readout.predict(test_features_np))
    if train_pred_np.ndim > 1:
        train_pred_np = np.argmax(train_pred_np, axis=-1)
    if test_pred_np.ndim > 1:
        test_pred_np = np.argmax(test_pred_np, axis=-1)

    val_labels_np = None
    val_pred_np = None
    if val_X is not None:
        val_labels_np = np.asarray(val_y)
        if val_labels_np.ndim > 1:
            val_labels_np = np.argmax(val_labels_np, axis=-1)
        val_features_np = runner.batch_transform(val_X, batch_size=feature_batch_size)
        if readout is None:
            val_pred_np = np.asarray(val_features_np)
        else:
            val_pred_np = np.asarray(readout.predict(val_features_np))
        if val_pred_np.ndim > 1:
            val_pred_np = np.argmax(val_pred_np, axis=-1)

    metrics_payload = dict(results.get("test", {})) if isinstance(results, dict) else {}
    plot_classification_results(
        train_labels=train_labels_np,
        test_labels=test_labels_np,
        train_predictions=train_pred_np,
        test_predictions=test_pred_np,
        val_labels=val_labels_np,
        val_predictions=val_pred_np,
        title=f"{model_type_str.upper()} on {dataset_name}",
        filename=filename,
        metrics_info=metrics_payload,
        best_lambda=selected_lambda,
        lambda_norm=lambda_norm,
    )


def _infer_filename_parts(topo_meta: Dict[str, Any], training_obj: Any, model_type_str: str) -> list[str]:
    filename_parts = [model_type_str, "raw"]
    feature_shape = None
    student_layers = None
    readout_label = None
    type_lower = str(model_type_str).lower()
    is_fnn = "fnn" in type_lower or "rnn" in type_lower or "nn" in type_lower
    has_reservoir = "reservoir" in type_lower or "distillation" in type_lower
    if isinstance(topo_meta, dict):
        shapes = topo_meta.get("shapes") or {}
        feature_shape = shapes.get("feature")
        details = topo_meta.get("details") or {}
        student_layers = details.get("student_layers")
        readout_label = details.get("readout")
        topo_type = str(topo_meta.get("type", "")).lower()
        is_fnn = is_fnn or "fnn" in topo_type or "rnn" in topo_type or "nn" in topo_type
        has_reservoir = has_reservoir or "reservoir" in topo_type or "distillation" in topo_type

    # Reservoir marker (nr) only if reservoir is involved
    if has_reservoir:
        if isinstance(feature_shape, tuple) and feature_shape:
            filename_parts.append(f"nr{int(feature_shape[0])}")
        else:
            filename_parts.append("nr0")

    # NN marker
    if is_fnn:
        layers = tuple(int(v) for v in student_layers) if student_layers is not None else ()
        if layers:
            filename_parts.append(f"nn{'-'.join(str(int(v)) for v in layers)}")
        else:
            filename_parts.append("nn0")
        filename_parts.append(f"epochs{int(getattr(training_obj, 'epochs', 0) or 0)}")
    return filename_parts


def generate_report(
    results: Dict[str, Any],
    config: Any,
    topo_meta: Dict[str, Any],
    *,
    runner: Any,
    readout: Any,
    train_X: Any,
    train_y: Any,
    test_X: Any,
    test_y: Any,
    val_X: Optional[Any],
    val_y: Optional[Any],
    training_obj: Any,
    dataset_name: str,
    model_type_str: str,
    task_type: Optional[Any] = None,
) -> None:
    # Loss plotting (distillation)
    training_logs = _safe_get(results, "training_logs", {})
    if training_logs:
        filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str)
        loss_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_loss.png"
        plot_distillation_loss(training_logs, loss_filename, title=f"{model_type_str.upper()} Distillation Loss")

    # Ridge search reporting
    train_res = _safe_get(results, "train", {})
    task_val = task_type if task_type is not None else getattr(config, "task_type", None)
    metric = "accuracy" if task_val and str(task_val).lower().find("class") != -1 else "mse"
    print_ridge_search_results(train_res, metric)

    # Classification plots
    if task_val and str(task_val).lower().find("class") != -1:
        filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str)
        confusion_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_confusion.png"
        selected_lambda = None
        lambda_norm = None
        if isinstance(train_res, dict):
            selected_lambda = train_res.get("best_lambda")
            weight_norms = train_res.get("weight_norms") or {}
            if selected_lambda is not None:
                lambda_norm = weight_norms.get(selected_lambda) or weight_norms.get(float(selected_lambda))
        plot_classification_report(
            runner=runner,
            readout=readout,
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            val_X=val_X,
            val_y=val_y,
            filename=confusion_filename,
            model_type_str=model_type_str,
            dataset_name=dataset_name,
            metric=metric,
            selected_lambda=selected_lambda,
            lambda_norm=lambda_norm,
            results=results,
            training_obj=training_obj,
        )
