"""
pipelines/run.py
Unified Pipeline Runner for JAX-based Models and Datasets.

V2 Architecture Compliance:
- Strict Configuration: No implicit defaults. Rely entirely on Presets + User Config.
- Canonical Names Only: No alias resolution (e.g., 'alpha' -> 'leak_rate') happens here.
- Fail Fast: Dictionary access raises KeyError if parameters are missing.
"""

from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Core Imports
from reservoir.models import FlaxModelFactory
from pipelines.generic_runner import UniversalPipeline
from reservoir.components import FeatureScaler, DesignMatrix, RidgeRegression, TransformerSequence
from reservoir.data.presets import DATASET_REGISTRY, DatasetPreset
from reservoir.data.registry import DatasetRegistry
from reservoir.models.orchestrator import ReservoirModel
from reservoir.models.presets import get_model_preset, ModelPreset
from reservoir.training.presets import get_training_preset, TrainingConfig
from reservoir.models.reservoir.classical import ClassicalReservoir

# Ensure dataset loaders are registered
from reservoir.data import loaders as _data_loaders  # noqa: F401


def _format_shape(shape: Optional[tuple[int, ...]]) -> str:
    if shape is None:
        return "None"
    if len(shape) == 1:
        return f"[{shape[0]}]"
    return f"[{'x'.join(str(d) for d in shape)}]"


def _print_topology(topo_meta: Optional[Dict[str, Any]]) -> None:
    if not topo_meta:
        return

    flow_parts = []
    shapes = topo_meta.get("shapes", {})
    typ = topo_meta.get("type", "").upper()
    details = topo_meta.get("details", {})

    print("=" * 40)
    print(f"Model Architecture: {typ or 'MODEL'}")
    print("=" * 40)

    input_shape = shapes.get("input")
    projected = shapes.get("projected")
    internal = shapes.get("internal")
    feature = shapes.get("feature")
    output = shapes.get("output")

    preprocess = details.get("preprocess") or "None (Pass-through)"
    agg_mode = details.get("agg_mode")
    student_layers = details.get("student_layers")

    print(f"1. Input Data      : { _format_shape(input_shape) }")
    print(f"2. Preprocessing   : { preprocess }")

    if typ == "FNN_DISTILLATION":
        flow_parts.append(_format_shape(input_shape))
        flow_parts.append(_format_shape(projected))
        if isinstance(internal, tuple):
            flat_shape = internal
            flow_parts.append(_format_shape(flat_shape))
        else:
            flat_shape = None
        hidden_str = "-".join(str(h) for h in (student_layers or [])) or "None"
        print(f"3. Input Projection: {_format_shape(input_shape)} -> {_format_shape(projected)} (time-distributed)")
        print(f"4. Student Model   : {_format_shape(flat_shape)} -> [{hidden_str}] -> {_format_shape(feature)}")
        print(f"5. Readout         : {_format_shape(feature)} -> {_format_shape(output)} outputs")
        flow_parts.append(f"[{hidden_str}]")
        flow_parts.append(_format_shape(feature))
        flow_parts.append(_format_shape(output))
    elif typ in {"RESERVOIR", "ESN", "CLASSICAL"}:
        flow_parts.append(_format_shape(input_shape))
        flow_parts.append(_format_shape(projected))
        flow_parts.append(_format_shape(internal))
        agg_desc = f"{_format_shape(internal)} -> {_format_shape(feature)}"
        if agg_mode:
            agg_desc += f" (mode={agg_mode})"
        print(f"3. Reservoir       : {_format_shape(projected)} -> {_format_shape(internal)} (Recurrent)")
        print(f"4. Aggregation     : {agg_desc}")
        print(f"5. Readout         : {_format_shape(feature)} -> {_format_shape(output)} outputs")
        flow_parts.append(_format_shape(feature))
        flow_parts.append(_format_shape(output))
    else:
        flow_parts.append(_format_shape(input_shape))
        flow_parts.append(_format_shape(internal))
        flow_parts.append(_format_shape(feature))
        flow_parts.append(_format_shape(output))
        print(f"3. Internal Struct : {_format_shape(internal)}")
        print(f"4. Readout         : {_format_shape(feature)} -> {_format_shape(output)} outputs")

    print("-" * 40)
    flow_str = " -> ".join(str(p) for p in flow_parts if p)
    if flow_str:
        print(f"Tensor Flow     : {flow_str}")
    print("=" * 40)


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
        "connectivity",
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


def load_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Public dataset loader used by pipelines.__getattr__."""
    X, y = _load_dataset(config)
    split_idx = int(0.8 * len(X))
    train_X, test_X = X[:split_idx], X[split_idx:]
    train_y, test_y = y[:split_idx], y[split_idx:]
    return {
        "train_X": train_X,
        "train_y": train_y,
        "test_X": test_X,
        "test_y": test_y,
    }


def _load_dataset(config: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load dataset via registry."""
    dataset_name = config.get("dataset")
    if not dataset_name:
        raise ValueError("Configuration Error: 'dataset' key missing in config.")

    normalized_name = DATASET_REGISTRY.normalize_name(dataset_name)
    loader = DatasetRegistry.get(normalized_name)
    if not loader:
        raise ValueError(f"Registry Error: No loader found for dataset '{normalized_name}'.")

    return loader(config)


def run_pipeline(
    config: Dict[str, Any],
    train_X: Optional[jnp.ndarray] = None,
    train_y: Optional[jnp.ndarray] = None,
    test_X: Optional[jnp.ndarray] = None,
    test_y: Optional[jnp.ndarray] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    The Unified Entry Point (V2 Strict Mode).
    """
    # --- 1. Data Preparation & Metadata Validation ---
    dataset_name, dataset_preset = _get_strict_dataset_meta(config)

    if train_X is None or train_y is None:
        print(f"Loading dataset: {dataset_name}...")
        X, y = _load_dataset(config)
        split_idx = int(0.8 * len(X))
        train_X, test_X = X[:split_idx], X[split_idx:]
        train_y, test_y = y[:split_idx], y[split_idx:]

    # Model type (required downstream)
    model_type = config.get("model_type")
    if not model_type:
        raise ValueError("Configuration Error: 'model_type' is required.")
    model_type = model_type.lower()

    # Classification/Shape Handling
    if dataset_name in {"mnist"}:
        num_classes = int(dataset_preset.config.n_output)
        print(f"Converting labels to one-hot vectors (classes={num_classes})...")
        train_y = jax.nn.one_hot(train_y.astype(int), num_classes)
        test_y = jax.nn.one_hot(test_y.astype(int), num_classes)
        print("Normalizing image data to [0, 1] range (div by 255)...")
        train_X = train_X.astype(jnp.float64) / 255.0
        if test_X is not None:
            test_X = test_X.astype(jnp.float64) / 255.0
        config["use_preprocessing"] = False

    # Determine Task Type
    preset_type = dataset_preset.task_type.lower()
    override_cls = config.get("is_classification")
    if override_cls is not None:
        is_classification = bool(override_cls)
    elif preset_type in {"classification", "regression"}:
        is_classification = preset_type == "classification"
    else:
        raise ValueError(f"Unknown preset task type: {preset_type}")

    meta_n_outputs = int(dataset_preset.config.n_output)

    # Resolve training configuration (needed for ridge search/validation)
    training_cfg = _resolve_training_config(config, is_classification)
    training_cfg["_meta_dataset"] = dataset_name
    training_cfg["_meta_model_type"] = model_type
    feature_batch_size = int(training_cfg.get("feature_batch_size", training_cfg.get("batch_size", 0) or 0))

    # Shape Adjustment Logic
    reservoir_units_for_plot: Optional[int] = None
    nn_hidden_for_plot: Optional[list[int]] = None

    if train_X.ndim != 3:
        raise ValueError(
            f"Model type '{model_type}' requires 3D input (Batch, Time, Features). "
            f"Got shape {train_X.shape}. Please reshape your data source."
        )


    print(f"Data Shapes -> Train: {train_X.shape}, Test: {test_X.shape if test_X is not None else 'None'}")
    input_shape = train_X.shape[1:]

    hidden_dim = config.get("hidden_dim")
    if hidden_dim is None:
        raise ValueError(f"Configuration Error: 'hidden_dim' must be specified.")

    # --- 2. Model Creation (Strict) ---
    print(f"Initializing {model_type.upper()} model via Factory...")

    if model_type in ["fnn", "rnn"]: #Distillation Modes (there are no standard FNN/RNN models in this codebase)
        model_cfg = dict(config.get("model", {}))

        # Enforce explicit dimensions or fail
        if model_type == "fnn":
            hidden_layers = model_cfg.get("layer_dims")
            if hidden_layers:
                hidden_layers = [int(v) for v in hidden_layers]
            else:
                hidden_layers = [int(hidden_dim)]

            model_cfg["layer_dims"] = [
                int(input_shape[-1]),
                *hidden_layers,
                int(meta_n_outputs)
            ]
            nn_hidden_for_plot = hidden_layers.copy() if hidden_layers else None

            teacher_cfg_candidate = config.get("reservoir") or config.get("reservoir_params") or {}
            teacher_units = teacher_cfg_candidate.get("n_units") or hidden_dim
            if teacher_units is not None:
                reservoir_units_for_plot = int(teacher_units)
        else:
            # RNN types


            model_cfg.setdefault("input_dim", int(input_shape[-1]))
            model_cfg["hidden_dim"] = int(hidden_dim)
            model_cfg.setdefault("output_dim", int(meta_n_outputs))
            model_cfg.setdefault("return_sequences", False)
            model_cfg.setdefault("return_hidden", False)

        factory_cfg = {
            "type": model_type,
            "model": model_cfg,
            "training": training_cfg,
            "reservoir": config.get("reservoir", {}),
            "input_dim": int(input_shape[-1]),
        }
        model = FlaxModelFactory.create_model(factory_cfg)

        raw_in = int(input_shape[-1])
        out_d = int(meta_n_outputs)
        feat_d = int(reservoir_units_for_plot) if reservoir_units_for_plot else (hidden_layers[-1] if hidden_layers else 0)

        t_steps = input_shape[0] if len(input_shape) > 1 else 1
        f_dim = raw_in
        flat_dim = t_steps * feat_d

        topo_meta = {
            "type": "FNN_DISTILLATION",
            "shapes": {
                "input": (t_steps, f_dim),
                "projected": (t_steps, feat_d),
                "internal": (flat_dim,),
                "feature": (feat_d,),
                "output": (out_d,),
            },
            "details": {
                "preprocess": None,
                "agg_mode": None,
                "student_layers": hidden_layers,
            },
        }



    elif model_type in ["reservoir", "esn", "classical"]:
        # Strict parameter resolution (Maps hidden_dim -> n_units)
        reservoir_params = _get_strict_reservoir_params(config)

        # Feature Engineering Config
        preprocess_steps = []
        effective_input_dim = int(input_shape[-1])

        if dataset_name not in {"mnist", "fashion_mnist"} and config.get("use_preprocessing", True):
            preprocess_steps.append(FeatureScaler())

        # Design Matrix
        use_dm = bool(reservoir_params.get("use_design_matrix", False))
        degree = reservoir_params.get("poly_degree")
        include_bias = reservoir_params.get("poly_bias", False)
        if use_dm:

            if degree is None:
                raise ValueError("Configuration Error: 'poly_degree' is required when 'use_design_matrix' is True.")

            print(f"Adding DesignMatrix (degree={degree}, bias={include_bias})...")
            preprocess_steps.append(DesignMatrix(degree=int(degree), include_bias=bool(include_bias)))

            factor = int(degree) if int(degree) > 0 else 1
            effective_input_dim = effective_input_dim * factor + (1 if include_bias else 0)

        preprocess = TransformerSequence(preprocess_steps) if preprocess_steps else None

        # Instantiate ClassicalReservoir
        # Dictionary access 'reservoir_params["key"]' ensures we fail fast if keys are missing.
        try:
            node = ClassicalReservoir(
                n_inputs=effective_input_dim,
                n_units=int(reservoir_params["n_units"]),  # Derived from hidden_dim if not explicit
                input_scale=float(reservoir_params["input_scale"]),
                spectral_radius=float(reservoir_params["spectral_radius"]),
                leak_rate=float(reservoir_params["leak_rate"]),
                connectivity=float(reservoir_params["connectivity"]),
                noise_rc=float(reservoir_params["noise_rc"]),
                bias_scale=float(reservoir_params["bias_scale"]),
                seed=int(reservoir_params["seed"]),
            )
        except KeyError as e:
            raise ValueError(
                f"Configuration Error: Missing required reservoir parameter {e}. "
                f"Preset: '{config.get('reservoir_preset', 'classical')}', Config: {reservoir_params}"
            )

        readout_mode = reservoir_params.get("state_aggregation")
        if not readout_mode:
            raise ValueError(
                "Configuration Error: 'state_aggregation' is missing in reservoir_params. "
                "Define it in the preset or override explicitly."
            )

        model = ReservoirModel(reservoir=node, preprocess=preprocess, readout_mode=readout_mode)
        reservoir_units_for_plot = int(node.n_units)

        raw_input_dim = int(input_shape[-1])
        res_units = int(node.n_units)
        out_dim = int(meta_n_outputs)
        prep_msg = None
        if preprocess:
            prep_msg = f"Expanded to {effective_input_dim} (PolyDegree={degree}, Bias={include_bias})"

        t_steps = input_shape[0] if len(input_shape) > 1 else 1
        f_dim = raw_input_dim
        agg_dim = res_units if str(readout_mode) != "flatten" else t_steps * res_units
        topo_meta = {
            "type": model_type.upper(),
            "shapes": {
                "input": (t_steps, f_dim),
                "projected": (t_steps, res_units),
                "internal": (t_steps, res_units),
                "feature": (agg_dim,),
                "output": (out_dim,),
            },
            "details": {
                "preprocess": prep_msg or "None (Pass-through)",
                "agg_mode": readout_mode,
                "student_layers": None,
            },
        }
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # --- Common Architecture Summary Block ---
    _print_topology(topo_meta)

    # --- 3a. Validation Split for ridge search ---
    val_tuple = None
    val_size = float(training_cfg.get("val_size", 0.0)) if training_cfg else 0.0
    if val_size > 0.0 and len(train_X) > 1:
        val_count = max(1, int(len(train_X) * val_size))
        train_count = len(train_X) - val_count
        if train_count < 1:
            train_count = len(train_X) - 1
        val_X = train_X[train_count:]
        val_y = train_y[train_count:]
        train_X = train_X[:train_count]
        train_y = train_y[:train_count]
        val_tuple = (val_X, val_y)


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
        validation=val_tuple,
        training_cfg=training_cfg,
    )

    filename_parts = [f"{dataset_name}", f"{model_type}", "raw"]
    if reservoir_units_for_plot is not None:
        filename_parts.append(f"nr{reservoir_units_for_plot}")
    if nn_hidden_for_plot:
        joined_nn = "-".join(str(v) for v in nn_hidden_for_plot)
        filename_parts.append(f"nn{joined_nn}")


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

    # --- 4. Persistence ---
    if save_path and hasattr(model, 'save'):
        print(f"Saving model to {save_path}...")
        model.save(save_path)

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
            if val_tuple is not None:
                val_X, val_y = val_tuple
                val_labels_np = np.asarray(val_y)
                if val_labels_np.ndim > 1:
                    val_labels_np = np.argmax(val_labels_np, axis=-1)
                val_features_np = runner.batch_transform(val_X, batch_size=feature_batch_size)
                val_pred_raw = np.asarray(readout.predict(val_features_np))
                val_pred_np = val_pred_raw
                if val_pred_np.ndim > 1:
                    val_pred_np = np.argmax(val_pred_np, axis=-1)


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
                metrics_info=results.get("test", {}),
            )

    return results

# --- Convenience Wrappers ---

def run_fnn_pipeline(config: Dict[str, Any], save_path: Optional[str] = None) -> Dict[str, Any]:
    config["model_type"] = "fnn"
    return run_pipeline(config, save_path=save_path)

def run_rnn_pipeline(config: Dict[str, Any], save_path: Optional[str] = None) -> Dict[str, Any]:
    config["model_type"] = "rnn"
    return run_pipeline(config, save_path=save_path)

def run_reservoir_pipeline(config: Dict[str, Any], save_path: Optional[str] = None) -> Dict[str, Any]:
    if "reservoir_type" not in config and "reservoir_preset" not in config:
        config["reservoir_preset"] = "classical"
    config["model_type"] = "reservoir"
    return run_pipeline(config, save_path=save_path)
