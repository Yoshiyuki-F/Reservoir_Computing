"""
pipelines/run.py
Unified Pipeline Runner for JAX-based Models and Datasets.

V2 Architecture Compliance:
- Strict Configuration: No implicit defaults. Rely entirely on Presets + User Config.
- Canonical Names Only: No alias resolution (e.g., 'alpha' -> 'leak_rate') happens here.
- Fail Fast: Dictionary access raises KeyError if parameters are missing.
"""

from dataclasses import replace
from typing import Dict, Any, Optional, Tuple

import numpy as np

# Core Imports
from reservoir.models import ModelFactory
from reservoir.models.distillation import DistillationModel
from reservoir.models.reservoir.model import ReservoirModel
from reservoir.models.nn.fnn import FNNModel
from reservoir.core.identifiers import Dataset, Pipeline, TaskType, RunConfig
from reservoir.utils.printing import print_topology
from reservoir.pipelines.generic_runner import UniversalPipeline
from reservoir.components import RidgeRegression
from reservoir.data.loaders import load_dataset_with_validation_split
from reservoir.data.presets import DATASET_REGISTRY, DatasetPreset
from reservoir.training.presets import get_training_preset, TrainingConfig

# Ensure dataset loaders are registered
from reservoir.data import loaders as _data_loaders  # noqa: F401




def _get_strict_dataset_meta(dataset: Dataset) -> Tuple[Dataset, DatasetPreset]:
    """
    Retrieve dataset metadata.
    Raises ValueError if the dataset is unknown or metadata is incomplete.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Configuration Error: 'dataset' must be a Dataset Enum, got {type(dataset)}.")

    preset = DATASET_REGISTRY.get(dataset)
    if preset is None:
        raise ValueError(f"Configuration Error: Dataset '{dataset}' is not registered in DATASET_REGISTRY.")

    if preset.config.n_input is None:
        raise ValueError(f"Preset Error: Dataset '{dataset}' is missing 'n_input' in its definition.")
    if preset.config.n_output is None:
        raise ValueError(f"Preset Error: Dataset '{dataset}' is missing 'n_output' in its definition.")

    return dataset, preset


def _resolve_training_config(task_type: TaskType) -> TrainingConfig:
    """
    Resolve training configuration strictly from presets.
    """
    preset = get_training_preset("standard")
    classification = task_type is TaskType.CLASSIFICATION
    return replace(preset, classification=classification)


def run_pipeline(config: RunConfig) -> Dict[str, Any]:
    """
    The Unified Entry Point (V2 Strict Mode).
    """
    if not isinstance(config, RunConfig):
        raise TypeError(f"run_pipeline requires RunConfig, got {type(config)}.")

    dataset_enum, dataset_preset = _get_strict_dataset_meta(config.dataset)
    dataset_name = dataset_enum.value
    task_type = config.task_type

    training_obj = _resolve_training_config(task_type)
    feature_batch_size = int(training_obj.batch_size or 0)

    train_X, train_y, val_X, val_y, test_X, test_y = load_dataset_with_validation_split(
        {"dataset": dataset_enum},
        dataset_preset,
        training_obj.to_dict(),
        model_type=config.model_type.value,
        require_3d=True,
    )

    input_shape = train_X.shape[1:]
    input_dim = int(input_shape[-1])

    def _flatten(arr: Optional[Any]) -> Optional[Any]:
        if arr is None:
            return None
        arr_np = np.asarray(arr)
        if arr_np.ndim == 3:
            return arr_np.reshape(arr_np.shape[0], -1)
        return arr

    if config.model_type is Pipeline.FNN:
        train_X = _flatten(train_X)
        val_X = _flatten(val_X)
        test_X = _flatten(test_X)
        input_shape = train_X.shape[1:]
        input_dim = int(input_shape[-1])

    if dataset_preset.config.n_output is not None:
        meta_n_outputs = int(dataset_preset.config.n_output)
    else:
        target_sample = train_y if train_y is not None else test_y
        if target_sample is None:
            raise ValueError("Unable to infer output dimension without targets.")
        meta_n_outputs = int(target_sample.shape[-1]) if hasattr(target_sample, "shape") and len(target_sample.shape) > 1 else 1

    print(f"Data Shapes -> Train: {train_X.shape}, Val: {getattr(val_X, 'shape', None)}, Test: {test_X.shape}")

    # Build model via Factory (all domain logic lives there)
    model = ModelFactory.create_model(
        config,
        dataset_preset=dataset_preset,
        training=training_obj,
        input_dim=input_dim,
        output_dim=meta_n_outputs,
    )

    # Flatten sequences for pure FNN students only (reservoir/distillation keep sequences)
    if isinstance(model, FNNModel):
        train_X = _flatten(train_X)
        val_X = _flatten(val_X)
        test_X = _flatten(test_X)
        input_shape = train_X.shape[1:]
        input_dim = int(input_shape[-1])

    model_type_str = config.model_type.value
    topo_meta: Optional[Dict[str, Any]] = None
    filename_res_units: Optional[int] = None
    filename_student_hidden: Optional[list[int]] = None

    t_steps = input_shape[0] if len(input_shape) > 1 else 1
    f_dim = int(input_shape[-1])

    if isinstance(model, DistillationModel):
        teacher_units = int(model.teacher_output_dim)
        student_hidden = [int(v) for v in model.student_hidden]
        projected_units = teacher_units
        internal_flat = t_steps * projected_units

        topo_meta = {
            "type": model_type_str.upper(),
            "shapes": {
                "input": (t_steps, f_dim),
                "projected": (t_steps, projected_units),
                "internal": (internal_flat,),
                "feature": (projected_units,),
                "output": (int(meta_n_outputs),),
            },
            "details": {
                "preprocess": None,
                "agg_mode": getattr(model, "teacher_config", None) and model.teacher_config.state_aggregation,
                "student_layers": student_hidden or None,
            },
        }
        filename_res_units = projected_units
        filename_student_hidden = student_hidden
    elif isinstance(model, ReservoirModel):
        res_units = int(getattr(model.reservoir, "n_units"))
        filename_res_units = res_units
        agg_mode = getattr(model, "readout_mode", None)
        topo_meta = {
            "type": model_type_str.upper(),
            "shapes": {
                "input": (t_steps, f_dim),
                "projected": (t_steps, res_units),
                "internal": (t_steps, res_units),
                "feature": (res_units,),
                "output": (int(meta_n_outputs),),
            },
            "details": {
                "preprocess": None,
                "agg_mode": agg_mode,
                "student_layers": None,
            },
        }
    elif isinstance(model, FNNModel):
        layer_dims = [int(v) for v in getattr(model, "layer_dims", [])]
        feature_units = layer_dims[-2] if len(layer_dims) >= 2 else None
        topo_meta = {
            "type": model_type_str.upper(),
            "shapes": {
                "input": (t_steps, f_dim),
                "projected": None,
                "internal": None,
                "feature": (feature_units,) if feature_units else None,
                "output": (int(meta_n_outputs),),
            },
            "details": {
                "preprocess": None,
                "agg_mode": None,
                "student_layers": layer_dims[1:-1] if len(layer_dims) > 2 else None,
            },
        }

    print_topology(topo_meta)

    readout = RidgeRegression(ridge_lambda=float(training_obj.ridge_lambda), use_intercept=True)

    metric = "accuracy" if task_type is TaskType.CLASSIFICATION else "mse"
    runner = UniversalPipeline(model, readout, None, metric=metric)
    validation_tuple = (val_X, val_y) if (val_X is not None and val_y is not None) else None
    results = runner.run(
        train_X,
        train_y,
        test_X,
        test_y,
        validation=validation_tuple,
        training_cfg=training_obj,
        training_extras={},
        model_label=model_type_str,
    )

    filename_parts = [f"{model_type_str}", "raw"]
    if filename_res_units is not None:
        filename_parts.append(f"nr{filename_res_units}")
    if filename_student_hidden:
        joined_nn = "-".join(str(v) for v in filename_student_hidden)
        filename_parts.append(f"nn{joined_nn}")
        filename_parts.append(f"epochs{training_obj.epochs}")

    training_logs = results.get("training_logs", {}) if isinstance(results, dict) else {}
    if isinstance(training_logs, dict) and training_logs.get("loss_history"):
        loss_history = training_logs["loss_history"]
        try:
            from reservoir.utils.plotting import plot_loss_history
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"Skipping distillation loss plotting due to import error: {exc}")
        else:
            loss_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_distillation_loss.png"
            plot_loss_history(loss_history, loss_filename, title=f"{model_type_str.upper()} Distillation Loss")

    train_res = results.get("train", {}) if isinstance(results, dict) else {}
    if isinstance(train_res, dict) and "search_history" in train_res and train_res.get("search_history"):
        history = train_res.get("search_history", {})
        best_lam = train_res.get("best_lambda")
        weight_norms = train_res.get("weight_norms", {})
        metric_label = "Accuracy" if metric == "accuracy" else "MSE"
        best_by_metric = max(history, key=history.get) if metric == "accuracy" and history else (
            min(history, key=history.get) if history else None
        )
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

    if task_type is TaskType.CLASSIFICATION and test_X is not None and test_y is not None:
        try:
            from reservoir.utils.plotting import plot_classification_results
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"Skipping plotting due to import error: {exc}")
        else:
            train_labels_np = np.asarray(train_y)
            test_labels_np = np.asarray(test_y)
            if train_labels_np.ndim > 1:
                train_labels_np = np.argmax(train_labels_np, axis=-1)
            if test_labels_np.ndim > 1:
                test_labels_np = np.argmax(test_labels_np, axis=-1)

            train_features_np = runner.batch_transform(train_X, batch_size=feature_batch_size)
            test_features_np = runner.batch_transform(test_X, batch_size=feature_batch_size)
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
                val_pred_raw = np.asarray(readout.predict(val_features_np))
                val_pred_np = val_pred_raw
                if val_pred_np.ndim > 1:
                    val_pred_np = np.argmax(val_pred_np, axis=-1)

            selected_lambda = None
            lambda_norm = None
            if isinstance(train_res, dict):
                selected_lambda = train_res.get("best_lambda")
                weight_norms = train_res.get("weight_norms") or {}
                if selected_lambda is not None:
                    lambda_norm = weight_norms.get(selected_lambda)
                    if lambda_norm is None:
                        lambda_norm = weight_norms.get(float(selected_lambda))
            if lambda_norm is None and hasattr(readout, "coef_"):
                coef_arr = getattr(readout, "coef_", None)
                if coef_arr is not None:
                    components = [np.asarray(coef_arr).ravel()]
                    intercept_arr = getattr(readout, "intercept_", None)
                    if intercept_arr is not None:
                        components.append(np.asarray(intercept_arr).ravel())
                    if components:
                        stacked = np.concatenate(components)
                        if stacked.size > 0:
                            lambda_norm = float(np.linalg.norm(stacked))

            metrics_payload = dict(results.get("test", {})) if isinstance(results, dict) else {}

            filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_confusion.png"
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

    return results
