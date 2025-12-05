"""
pipelines/run.py
Unified Pipeline Runner for JAX-based Models and Datasets.

V2 Architecture Compliance:
- Strict Configuration: No implicit defaults. Rely entirely on Presets + User Config.
- Canonical Names Only: No alias resolution (e.g., 'alpha' -> 'leak_rate') happens here.
- Fail Fast: Dictionary access raises KeyError if parameters are missing.
"""

from typing import Dict, Any, Optional, Tuple

import numpy as np

# Core Imports
from reservoir.models import ModelFactory
from reservoir.models.distillation import DistillationModel
from reservoir.models.reservoir.model import ReservoirModel
from reservoir.models.nn.fnn import FNNModel
from reservoir.utils.printing import print_topology
from pipelines.generic_runner import UniversalPipeline
from reservoir.components import RidgeRegression
from reservoir.data.loaders import load_dataset_with_validation_split
from reservoir.data.presets import DATASET_REGISTRY, DatasetPreset
from reservoir.models.presets import get_model_preset, ModelPreset
from reservoir.training.presets import get_training_preset, TrainingConfig

# Ensure dataset loaders are registered
from reservoir.data import loaders as _data_loaders  # noqa: F401




def _get_strict_dataset_meta(config: Dict[str, Any]) -> Tuple[str, DatasetPreset]:
    """
    Retrieve dataset metadata.
    Raises ValueError if the dataset is unknown or metadata is incomplete.
    """
    name = config.get("dataset")
    if not name:
        raise ValueError("Configuration Error: 'dataset' name is missing.")

    normalized_name = DATASET_REGISTRY.normalize_name(name)
    preset = DATASET_REGISTRY.get(normalized_name)

    if preset is None:
        raise ValueError(f"Configuration Error: Dataset '{name}' is not registered in DATASET_REGISTRY.")

    if preset.config.n_output is None:
        raise ValueError(f"Preset Error: Dataset '{name}' is missing 'n_output' in its definition.")

    return normalized_name, preset


def _resolve_training_config(config: Dict[str, Any], is_classification: bool) -> Dict[str, Any]:
    """
    Resolve training configuration strictly.
    """
    preset_name = config.get("training_preset", "standard")
    try:
        preset: TrainingConfig = get_training_preset(preset_name)
    except KeyError:
        raise ValueError(f"Configuration Error: Training preset '{preset_name}' not found.")

    overrides = dict(config.get("training", {}) or {})
    if "alpha" in overrides:
        raise ValueError("Ambiguous parameter 'alpha' is forbidden. Use 'ridge_lambda' (readout) or 'leak_rate' (reservoir).")
    preset_dict = preset.to_dict()

    # Split overrides into validated dataclass fields vs. pass-through extras.
    allowed_keys = set(TrainingConfig.__dataclass_fields__.keys())
    field_overrides = {k: v for k, v in overrides.items() if k in allowed_keys}
    passthrough = {k: v for k, v in overrides.items() if k not in allowed_keys}

    merged_fields = {**preset_dict, **field_overrides}
    try:
        validated_cfg = TrainingConfig(**merged_fields)  # Validation happens in __post_init__
    except TypeError as exc:
        raise ValueError(f"Configuration Error: Invalid training override keys. {exc}")

    merged = validated_cfg.to_dict()
    merged.update(passthrough)

    if is_classification:
        merged["classification"] = True
    return merged


def _get_strict_reservoir_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge ModelPreset parameters with User Configuration.
    Maps generic CLI args (hidden_dim) to model specific args (n_units).
    """
    # 1. Identify Preset
    preset_name = config.get("reservoir_preset") or config.get("reservoir_type")
    if not preset_name:
        preset_name = "classical"

    try:
        preset: ModelPreset = get_model_preset(preset_name)
    except KeyError:
        raise ValueError(f"Configuration Error: Model Preset '{preset_name}' not found.")

    # 2. Base params from Preset (Canonical names only)
    base_params = preset.to_params()

    # 3. Overrides from User (Canonical names only expected)
    user_overrides = dict(config.get("reservoir", {}) or {})

    # Map generic CLI 'hidden_dim' to 'n_units' (explicit override)
    if "hidden_dim" in config and config["hidden_dim"] is not None:
        user_overrides["n_units"] = config["hidden_dim"]

    # 4. Merge
    merged = {**base_params, **user_overrides}

    required = [
        "n_units",
        "input_scale",
        "spectral_radius",
        "leak_rate",
        "input_connectivity",
        "rc_connectivity",
        "bias_scale",
        "noise_rc",
        "seed",
    ]
    missing = [key for key in required if key not in merged or merged[key] is None]
    if missing:
        raise ValueError(
            f"Configuration Error: Missing required reservoir parameters {missing}. "
            "Provide them via preset or explicit overrides."
        )

    return merged

def run_pipeline(
    config: Dict[str, Any],
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    The Unified Entry Point (V2 Strict Mode).
    """
    # --- 1. loading presets ---
    dataset_name: Optional[str] = None
    dataset_preset: Optional[DatasetPreset] = None
    provided = data or {}
    provided_splits = (
        provided.get("train_X"),
        provided.get("train_y"),
        provided.get("test_X"),
        provided.get("test_y"),
    )
    has_provided_data = all(split is not None for split in provided_splits)

    if config.get("dataset"):
        dataset_name, dataset_preset = _get_strict_dataset_meta(config)

    preset_type = dataset_preset.task_type.lower() if dataset_preset else None
    override_cls = config.get("is_classification")
    if override_cls is not None:
        is_classification = bool(override_cls)
    elif preset_type in {"classification", "regression"}:
        is_classification = preset_type == "classification"
    else:
        is_classification = bool(config.get("classification", False))

    # Model type (required downstream)
    model_type = config.get("model_type")
    if not model_type:
        raise ValueError("Configuration Error: 'model_type' is required.")
    model_type = model_type.lower()

    # Resolve training configuration (needed for ridge search/validation)
    training_cfg = _resolve_training_config(config, is_classification)
    training_cfg["_meta_dataset"] = dataset_name
    training_cfg["_meta_model_type"] = model_type
    feature_batch_size = int(training_cfg.get("feature_batch_size", training_cfg.get("batch_size", 0) or 0))
    training_obj = TrainingConfig(
        **{
            key: training_cfg[key]
            for key in TrainingConfig.__dataclass_fields__.keys()
            if key in training_cfg
        }
    )
    if has_provided_data:
        train_X = provided["train_X"]
        train_y = provided["train_y"]
        test_X = provided["test_X"]
        test_y = provided["test_y"]
        val_X = provided.get("val_X")
        val_y = provided.get("val_y")
    else:
        if dataset_preset is None:
            raise ValueError("Configuration Error: dataset must be specified when data splits are not provided.")
        train_X, train_y, val_X, val_y, test_X, test_y = load_dataset_with_validation_split(
            config,
            dataset_preset,
            training_cfg,
            model_type=model_type,
            require_3d=True,
        )

    # Flatten for pure FNN (non-distillation) so the student sees vector inputs.
    is_distillation_intent = model_type == "fnn" and bool(config.get("reservoir") or config.get("reservoir_params"))
    if model_type == "fnn" and not is_distillation_intent:
        def _flatten(arr: Optional[Any]) -> Optional[Any]:
            if arr is None:
                return None
            arr_np = np.asarray(arr)
            if arr_np.ndim == 3:
                return arr_np.reshape(arr_np.shape[0], -1)
            return arr

        train_X = _flatten(train_X)
        val_X = _flatten(val_X)
        test_X = _flatten(test_X)

    print(f"Data Shapes -> Train: {train_X.shape}, Val: {getattr(val_X, 'shape', None)}, Test: {test_X.shape}")

    preset_input_dim: Optional[int] = None
    preset_output_dim: Optional[int] = None
    if dataset_preset is not None:
        if dataset_preset.config.n_input:
            preset_input_dim = int(dataset_preset.config.n_input)
        if dataset_preset.config.n_output:
            preset_output_dim = int(dataset_preset.config.n_output)

    if preset_output_dim is not None:
        meta_n_outputs = preset_output_dim
    else:
        target_sample = train_y if train_y is not None else test_y
        if target_sample is None:
            raise ValueError("Unable to infer output dimension without targets.")
        meta_n_outputs = int(target_sample.shape[-1]) if hasattr(target_sample, "shape") and len(target_sample.shape) > 1 else 1

    input_shape = train_X.shape[1:]
    input_dim_for_model = int(input_shape[-1])
    if preset_input_dim is not None and is_distillation_intent:
        input_dim_for_model = preset_input_dim

    hidden_dim = config.get("hidden_dim")
    if hidden_dim is None and model_type in ["reservoir", "classical", "esn"]:
        raise ValueError("Configuration Error: 'hidden_dim' must be specified for reservoir models.")

    model_cfg = dict(config.get("model", {}) or {})
    if model_type == "fnn":
        hidden_layers = model_cfg.get("layer_dims") or config.get("nn_hidden") or []
        model_cfg["layer_dims"] = [
            input_dim_for_model,
            *[int(v) for v in hidden_layers],
            int(preset_output_dim if preset_output_dim is not None else meta_n_outputs),
        ]
    elif model_type == "rnn":
        if "input_dim" not in model_cfg:
            model_cfg["input_dim"] = input_dim_for_model
        if "output_dim" not in model_cfg and preset_output_dim is not None:
            model_cfg["output_dim"] = int(preset_output_dim)

    # --- 2. Model Creation (Strict & Simplified) ---
    print(f"Initializing {model_type.upper()} model via Factory...")

    factory_cfg = {
        "type": model_type,
        "input_dim": int(input_dim_for_model),
        "model": model_cfg,
        "training": training_obj,
        "reservoir": _get_strict_reservoir_params(config) if model_type in ["reservoir", "classical", "esn"] else config.get("reservoir"),
        "reservoir_params": config.get("reservoir_params"),
        "use_preprocessing": config.get("use_preprocessing", True),
        "dataset": dataset_name,
        "hidden_dim": config.get("hidden_dim"),
    }

    model = ModelFactory.create_model(factory_cfg)

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
            "type": "FNN_DISTILLATION",
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
            "type": model_type.upper(),
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
            "type": model_type.upper(),
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

    # --- Common Architecture Summary Block ---
    print_topology(topo_meta)

    # --- 3a. Validation Split for ridge search ---
    # Shared Readout (owned by runner)
    readout_cfg = dict(config.get("readout", {}) or {})
    ridge_lambda = readout_cfg.get("ridge_lambda", training_cfg["ridge_lambda"])

    fit_intercept_cfg = readout_cfg.get("fit_intercept")
    use_intercept = True if fit_intercept_cfg is None else bool(fit_intercept_cfg)

    readout = RidgeRegression(ridge_lambda=float(ridge_lambda), use_intercept=use_intercept)

    # --- 3. Execution ---
    metric = "accuracy" if is_classification else "mse"
    runner = UniversalPipeline(model, readout, config.get("save_path"), metric=metric)
    results = runner.run(
        train_X,
        train_y,
        test_X,
        test_y,
        validation=(val_X, val_y),
        training_cfg=training_cfg,
    )

    filename_parts = [f"{model_type}", "raw"]
    if filename_res_units is not None:
        filename_parts.append(f"nr{filename_res_units}")
    if filename_student_hidden:
        joined_nn = "-".join(str(v) for v in filename_student_hidden)
        filename_parts.append(f"nn{joined_nn}")
        filename_parts.append(f"epochs{training_cfg.get('epochs')}")


    # --- 4a. Distillation Loss Visualization (Phase 1) ---
    training_logs = results.get("training_logs", {}) if isinstance(results, dict) else {}
    if isinstance(training_logs, dict) and training_logs.get("loss_history"):
        loss_history = training_logs["loss_history"]
        try:
            from pipelines.plotting import plot_loss_history
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"Skipping distillation loss plotting due to import error: {exc}")
        else:
            loss_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_distillation_loss.png"
            plot_loss_history(loss_history, loss_filename, title=f"{model_type.upper()} Distillation Loss")


    # Pull fitted readout (runner-owned)
    if isinstance(results, dict) and "readout" in results:
        readout = results["readout"]

    # Log ridge search results if available
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

    # --- 5. Visualization (classification only) ---
    if is_classification and test_X is not None and test_y is not None:
        try:
            from pipelines.plotting import plot_classification_results
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
                title=f"{model_type.upper()} on {dataset_name}",
                filename=filename,
                metrics_info=metrics_payload,
                best_lambda=selected_lambda,
                lambda_norm=lambda_norm,
            )

    return results
