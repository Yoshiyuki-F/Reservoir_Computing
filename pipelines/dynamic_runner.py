"""Dynamic experiment orchestration utilities for any model type."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from core_lib.core import ConfigComposer, ExperimentConfig
from core_lib.data import (
    ExperimentDataset,
    ExperimentDatasetClassification,
    prepare_experiment_data,
)
from core_lib.utils import calculate_mae, calculate_mse, denormalize_data
from pipelines.builders import build_fnn_config, build_reservoir_model
from pipelines.dispatchers import CLASSIFICATION_PIPELINES, REGRESSION_PIPELINES
from pipelines.experiment_utils import (
    _log_ridge_search,
    _resolve_model_kind,
    _save_config_snapshot,
    finalize_classification_report,
)
from pipelines.fnn_pipeline import run_reservoir_emulation_pipeline
from pipelines.naming import resolve_experiment_naming
from pipelines.plotting import plot_prediction_results


def run_dynamic_experiment(
    dataset_name: str,
    model_name: str,
    training_name: str = "standard",
    show_training: bool = False,
    backend: Optional[str] = None,
    n_hidden_layer_override: Optional[int] = None,
    comparison_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], float, Optional[float], float]:
    """Run an experiment with dynamic configuration."""

    # Fast-path: FNN vs Reservoir emulation comparison
    if comparison_config is not None:
        print("=== Dynamic Experiment: FNN vs Reservoir (Emulation) ===")
        res_size = int(comparison_config["reservoir_size"])
        hidden_candidates = [
            comparison_config.get("n_hidden"),
            comparison_config.get("n_hidden_layer"),
            comparison_config.get("fnn_hidden"),
        ]
        n_hidden: Optional[int] = None
        for candidate in hidden_candidates:
            if candidate is None:
                continue
            try:
                n_hidden = int(candidate)
                break
            except (TypeError, ValueError):
                continue
        if n_hidden is None:
            n_hidden = 100
            print("Warning: n_hidden for FNN comparison was not provided; defaulting to 100.")

        fnn_config = build_fnn_config(
            dataset_name=dataset_name,
            n_hidden=n_hidden,
            reservoir_size=res_size,
            training_name=training_name or "standard",
        )
        results = run_reservoir_emulation_pipeline(
            fnn_config,
            reservoir_size=res_size,
            time_steps=28,  # MNIST fixed
            backend=backend or "cpu",
        )
        metrics_for_plot = {
            "train_mse": float(results["train_mse"]),
            "test_mse": float(results["test_mse"]),
            "train_accuracy": float(results["train_accuracy"]),
            "test_accuracy": float(results["test_accuracy"]),
        }
        train_pred = np.asarray(results.get("train_predictions", []))
        test_pred = np.asarray(results.get("test_predictions", []))
        train_labels = np.asarray(results.get("train_labels", []))
        test_labels = np.asarray(results.get("test_labels", []))

        hidden_dim: Optional[int] = None
        model_cfg = getattr(fnn_config, "model", None)
        if model_cfg is not None and getattr(model_cfg, "hidden_dims", None):
            try:
                hidden_dim = int(model_cfg.hidden_dims[0])
            except (TypeError, ValueError):
                hidden_dim = None

        suffix_parts = []
        if hidden_dim is not None:
            suffix_parts.append(f"h{hidden_dim}")
        suffix_parts.append(f"vs_res{res_size}")
        suffix_segment = "_".join(suffix_parts)
        dataset_dir = Path(dataset_name).name
        base_name = f"{dataset_dir}_fnn_raw_{suffix_segment}_emulation"
        base_dir = Path("outputs") / dataset_dir
        base_dir.mkdir(parents=True, exist_ok=True)

        plot_title = f"FNN Reservoir Emulation (h={hidden_dim or 'unknown'}, N={res_size})"
        base_output = base_dir / f"{base_name}.png"
        snapshot_data = {
            "dataset": dataset_name,
            "model": "fnn_reservoir_emulation",
            "params": {
                "reservoir_size": res_size,
                "fnn_config": fnn_config.model_dump() if hasattr(fnn_config, "model_dump") else getattr(fnn_config, "__dict__", fnn_config),
            },
            "results": {
                "metrics": metrics_for_plot,
            },
        }
        finalize_classification_report(
            output_filename=base_output,
            plot_title=plot_title,
            metrics=metrics_for_plot,
            train_labels=train_labels.astype(int, copy=False),
            test_labels=test_labels.astype(int, copy=False),
            train_pred=train_pred.astype(int, copy=False),
            test_pred=test_pred.astype(int, copy=False),
            snapshot_payload=snapshot_data,
        )
        return (
            float(results["train_mse"]),
            float(results["test_mse"]),
            float(results["train_accuracy"]),
            float(results["test_accuracy"]),
        )

    # Create dynamic experiment configuration
    dynamic_experiment = {
        "name": f"{dataset_name}_{model_name}_{training_name}",
        "description": f"Dynamic experiment: {dataset_name} with {model_name} using {training_name} training",
        "dataset": dataset_name,
        "model": model_name,
        "training": training_name,
        "visualization": {
            "show_training": show_training
        }
    }

    composer = ConfigComposer()
    composed_config = composer.compose_experiment(dynamic_experiment)
    legacy_config = composer.compose_legacy_format(composed_config)

    model_name_lower = model_name.lower()
    is_quantum_choice = "quantum" in model_name_lower
    if not is_quantum_choice and n_hidden_layer_override is None:
        raise ValueError(
            "n_hidden_layer_override is required for classical reservoir models. "
            "Pass --n-hidden_layer via CLI or script."
        )

    if n_hidden_layer_override is not None:
        try:
            override_value = int(n_hidden_layer_override)
        except (TypeError, ValueError) as exc:
            raise ValueError("n_hidden_layer_override must be an integer") from exc

        model_section = legacy_config.get("model", {})
        model_type = str(model_section.get("model_type", "reservoir")).lower()
        if "quantum" not in model_type:
            reservoir_section = legacy_config.get("reservoir")
            if isinstance(reservoir_section, dict):
                reservoir_section["n_hidden_layer"] = override_value
            params = model_section.setdefault("params", {})
            if isinstance(params, dict):
                params["n_hidden_layer"] = override_value

    if dataset_name == 'mnist':
        training_cfg = legacy_config.get("training", {}).copy()
        training_cfg["task_type"] = "classification"
        legacy_config["training"] = training_cfg

    experiment_config = ExperimentConfig(**legacy_config)
    is_quantum = "quantum" in model_name or experiment_config.model.model_type == "quantum"
    dataset = prepare_experiment_data(experiment_config)

    return run_experiment(
        experiment_config,
        dataset,
        backend=backend,
        quantum_mode=is_quantum,
    )


def run_experiment(
    demo_config: ExperimentConfig,
    dataset: Any,
    backend: Optional[str] = None,
    quantum_mode: bool = False,
) -> Tuple[Optional[float], float, Optional[float], float]:
    """Execute training, prediction, and reporting for a prepared dataset."""

    print(f"=== {demo_config.demo.title} ===")

    build_result = build_reservoir_model(
        demo_config,
        dataset,
        backend=backend,
        quantum_mode=quantum_mode,
    )
    rc = build_result.rc
    reservoir_info = build_result.reservoir_info
    model_type = build_result.model_type
    dataset_name = demo_config.data_generation.name
    model_name = demo_config.model.name

    output_filename, plot_title = resolve_experiment_naming(
        demo_config,
        rc,
        reservoir_info,
        dataset_name=dataset_name,
        model_type=model_type,
        quantum_mode=quantum_mode,
        is_quantum_model=build_result.is_quantum_model,
        raw_training=build_result.raw_training,
        n_hidden_layer=build_result.n_hidden_layer,
    )
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    ridge_cfg = demo_config.training.ridge_lambdas
    if ridge_cfg and isinstance(ridge_cfg, (list, tuple)) and len(ridge_cfg) == 3:
        start, stop, num = ridge_cfg
        lambda_candidates = list(np.logspace(start, stop, int(num)))
    elif ridge_cfg:
        lambda_candidates = list(ridge_cfg)
    else:
        lambda_candidates = [1e-6, 1e-5, 1e-4, 1e-3]

    if isinstance(dataset, ExperimentDatasetClassification):
        return _run_classification_routine(
            rc,
            dataset,
            model_name=model_name,
            model_type=model_type or "",
            lambda_candidates=lambda_candidates,
            output_filename=output_filename,
            plot_title=plot_title,
            demo_config=demo_config,
            reservoir_info=reservoir_info,
        )

    return _run_regression_routine(
        rc,
        dataset,
        lambda_candidates=lambda_candidates,
        plot_title=plot_title,
        output_filename=output_filename,
        demo_config=demo_config,
        model_name=model_name,
        model_type=model_type or "",
    )


def _run_classification_routine(
    rc: Any,
    dataset: ExperimentDatasetClassification,
    *,
    model_name: str,
    model_type: str,
    lambda_candidates: Sequence[float],
    output_filename: str,
    plot_title: str,
    demo_config: ExperimentConfig,
    reservoir_info: Dict[str, Any],
) -> Tuple[float, float, float, float]:
    train_count = int(dataset.train_sequences.shape[0])
    val_count = int(dataset.val_sequences.shape[0])
    test_count = int(dataset.test_sequences.shape[0])
    print(f"データ分割 → train: {train_count}, val: {val_count}, test: {test_count}")
    print("訓練中 (classification)...")

    model_kind = _resolve_model_kind(model_name, model_type)
    class_train_fn, class_predict_fn = CLASSIFICATION_PIPELINES.get(
        model_kind,
        CLASSIFICATION_PIPELINES["classical"],
    )

    train_features = class_train_fn(
        rc,
        dataset.train_sequences,
        dataset.train_labels,
        lambda_candidates,
        10,
        True,
    )
    ridge_log = _log_ridge_search(rc)

    print("予測中 (train/test/val inference)...")

    def _predict_with_cache(
        sequences,
        *,
        cached_features,
        desc: str,
    ):
        return class_predict_fn(rc, sequences, cached_features, desc)

    train_logits = _predict_with_cache(
        dataset.train_sequences,
        cached_features=train_features,
        desc="Encoding train eval sequences",
    )
    test_logits = class_predict_fn(
        rc,
        dataset.test_sequences,
        None,
        "Encoding test sequences",
    )
    val_logits = None
    if hasattr(dataset, "val_sequences") and dataset.val_sequences.size > 0:
        val_logits = class_predict_fn(
            rc,
            dataset.val_sequences,
            None,
            "Encoding validation sequences",
        )

    train_one_hot = jnp.zeros((dataset.train_labels.shape[0], 10), dtype=jnp.float64)
    train_one_hot = train_one_hot.at[jnp.arange(dataset.train_labels.shape[0]), dataset.train_labels].set(1.0)
    test_one_hot = jnp.zeros((dataset.test_labels.shape[0], 10), dtype=jnp.float64)
    test_one_hot = test_one_hot.at[jnp.arange(dataset.test_labels.shape[0]), dataset.test_labels].set(1.0)
    if val_logits is not None:
        val_one_hot = jnp.zeros((dataset.val_labels.shape[0], 10), dtype=jnp.float64)
        val_one_hot = val_one_hot.at[jnp.arange(dataset.val_labels.shape[0]), dataset.val_labels].set(1.0)
    else:
        val_one_hot = None

    train_mse = float(calculate_mse(train_logits, train_one_hot))
    test_mse = float(calculate_mse(test_logits, test_one_hot))
    val_mse = float(calculate_mse(val_logits, val_one_hot)) if val_logits is not None and val_one_hot is not None else None

    train_pred = jnp.argmax(train_logits, axis=1)
    test_pred = jnp.argmax(test_logits, axis=1)
    train_accuracy = float(jnp.mean(train_pred == dataset.train_labels))
    test_accuracy = float(jnp.mean(test_pred == dataset.test_labels))
    if val_logits is not None:
        val_pred = jnp.argmax(val_logits, axis=1)
        val_accuracy = float(jnp.mean(val_pred == dataset.val_labels))
    else:
        val_pred = None
        val_accuracy = None

    accuracy_msg = f"Accuracy: train={train_accuracy:.4f}"
    if val_accuracy is not None:
        accuracy_msg += f", val={val_accuracy:.4f}"
    accuracy_msg += f", test={test_accuracy:.4f}"
    print(accuracy_msg)

    current_metrics: Dict[str, Any] = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }
    if val_mse is not None:
        current_metrics["val_mse"] = val_mse
    if val_accuracy is not None:
        current_metrics["val_accuracy"] = val_accuracy

    best_lambda_attr = getattr(rc, "best_ridge_lambda", None)
    if best_lambda_attr is not None:
        best_lambda = float(best_lambda_attr)
        current_metrics["best_ridge_lambda"] = best_lambda
    else:
        best_lambda = None

    extra_results = {
        "classification": {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "reservoir_info": reservoir_info,
        }
    }
    finalize_classification_report(
        output_filename=output_filename,
        plot_title=plot_title,
        metrics=current_metrics,
        train_labels=np.asarray(dataset.train_labels),
        test_labels=np.asarray(dataset.test_labels),
        train_pred=np.asarray(train_pred),
        test_pred=np.asarray(test_pred),
        val_labels=np.asarray(dataset.val_labels) if hasattr(dataset, "val_labels") and dataset.val_labels.size > 0 else None,
        val_pred=np.asarray(val_pred) if val_pred is not None else None,
        ridge_lambda=best_lambda,
        ridge_log=ridge_log,
        config=demo_config,
        extra_results=extra_results,
    )

    return train_mse, test_mse, train_accuracy, test_accuracy


def _run_regression_routine(
    rc: Any,
    dataset: ExperimentDataset,
    *,
    lambda_candidates: Sequence[float],
    plot_title: str,
    output_filename: str,
    demo_config: ExperimentConfig,
    model_name: str,
    model_type: str,
) -> Tuple[Optional[float], float, Optional[float], float]:
    print("訓練中...")
    model_kind = _resolve_model_kind(model_name, model_type)
    train_fn, predict_fn = REGRESSION_PIPELINES.get(
        model_kind,
        REGRESSION_PIPELINES["classical"],
    )
    train_fn(
        rc,
        dataset.train_input,
        dataset.train_target,
        lambda_candidates,
    )

    ridge_log = _log_ridge_search(rc)

    def _predict_and_denormalize(inputs: jnp.ndarray, targets: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        preds_norm = predict_fn(rc, inputs)
        preds = denormalize_data(preds_norm, dataset.target_mean, dataset.target_std)
        target_slice = targets
        if target_slice.shape[0] > preds.shape[0]:
            start = target_slice.shape[0] - preds.shape[0]
            target_slice = target_slice[start:, ...]
        elif target_slice.shape[0] < preds.shape[0]:
            raise ValueError(
                f"Prediction length exceeds target length ({preds.shape[0]} vs {target_slice.shape[0]})"
            )
        target_orig = denormalize_data(target_slice, dataset.target_mean, dataset.target_std)
        return preds, target_orig

    print("予測中...")
    test_predictions, test_target_orig = _predict_and_denormalize(
        dataset.test_input, dataset.test_target
    )

    test_mse = calculate_mse(test_predictions, test_target_orig)
    test_mae = calculate_mae(test_predictions, test_target_orig)

    train_mse = train_mae = None
    train_predictions_orig = train_target_orig = None

    if dataset.train_input.size > 0:
        train_predictions_orig, train_target_orig = _predict_and_denormalize(
            dataset.train_input, dataset.train_target
        )

        train_mse = calculate_mse(train_predictions_orig, train_target_orig)
        train_mae = calculate_mae(train_predictions_orig, train_target_orig)

        if demo_config.demo.show_training:
            print(f"訓練 MSE: {train_mse:.6f}, MAE: {train_mae:.6f}")

    print(f"テスト MSE: {test_mse:.6f}, MAE: {test_mae:.6f}")

    time_indices = np.arange(test_target_orig.shape[0])

    metrics_caption = {}
    metrics_snapshot = {}
    if train_mse is not None:
        metrics_caption["Train MSE"] = float(train_mse)
        metrics_snapshot["train_mse"] = float(train_mse)
    if train_mae is not None:
        metrics_caption["Train MAE"] = float(train_mae)
        metrics_snapshot["train_mae"] = float(train_mae)
    metrics_caption["Test MSE"] = float(test_mse)
    metrics_caption["Test MAE"] = float(test_mae)
    metrics_snapshot["test_mse"] = float(test_mse)
    metrics_snapshot["test_mae"] = float(test_mae)

    best_lambda_attr = getattr(rc, "best_ridge_lambda", None)
    if best_lambda_attr is not None:
        best_lambda = float(best_lambda_attr)
        metrics_caption["Ridge λ"] = f"{best_lambda:.2e}"
        metrics_snapshot["best_ridge_lambda"] = best_lambda

    plot_prediction_results(
        np.asarray(test_target_orig),
        np.asarray(test_predictions),
        np.asarray(time_indices),
        plot_title,
        output_filename,
        np.asarray(train_target_orig) if train_target_orig is not None else None,
        np.asarray(train_predictions_orig) if train_predictions_orig is not None else None,
        dataset.train_size if dataset.train_input.size > 0 else None,
        y_axis_label=demo_config.demo.y_axis_label,
        metrics_info=metrics_caption,
        add_test_zoom=demo_config.demo.add_test_zoom,
        zoom_range=demo_config.demo.zoom_range,
    )

    _save_config_snapshot(demo_config, output_filename, metrics_snapshot, ridge_log)

    return train_mse, test_mse, train_mae, test_mae
