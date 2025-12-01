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
from reservoir.models.presets import MODEL_REGISTRY, get_model_preset, ModelPreset
from reservoir.training.presets import get_training_preset, TrainingConfig
from reservoir.models.reservoir.classical import ClassicalReservoir

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

    merged = preset.to_dict()
    overrides = dict(config.get("training", {}) or {})
    merged.update(overrides)

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

    # Classification/Shape Handling
    if dataset_name in {"mnist", "fashion_mnist"}:
        num_classes = int(dataset_preset.config.n_output)
        print(f"Converting labels to one-hot vectors (classes={num_classes})...")
        train_y = jax.nn.one_hot(train_y.astype(int), num_classes)
        test_y = jax.nn.one_hot(test_y.astype(int), num_classes)
        print("Normalizing image data to [0, 1] range (div by 255)...")
        train_X = train_X.astype(jnp.float32) / 255.0
        if test_X is not None:
            test_X = test_X.astype(jnp.float32) / 255.0
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

    # Shape Adjustment Logic
    model_type = config.get("model_type")
    if not model_type:
        raise ValueError("Configuration Error: 'model_type' is required.")
    model_type = model_type.lower()

    if model_type == "fnn":
        if train_X.ndim > 2:
            print(f"Flattening input for FNN: {train_X.shape} -> (N, Flattened)")
            train_X = train_X.reshape(train_X.shape[0], -1)
            if test_X is not None:
                test_X = test_X.reshape(test_X.shape[0], -1)

    elif model_type in ["rnn", "reservoir", "esn", "classical"]:
        if train_X.ndim != 3:
            raise ValueError(
                f"Model type '{model_type}' requires 3D input (Batch, Time, Features). "
                f"Got shape {train_X.shape}. Please reshape your data source."
            )

    print(f"Data Shapes -> Train: {train_X.shape}, Test: {test_X.shape if test_X is not None else 'None'}")
    input_shape = train_X.shape[1:]

    # --- 2. Model Creation (Strict) ---
    print(f"Initializing {model_type.upper()} model via Factory...")

    if model_type in ["fnn", "rnn", "lstm", "gru"]:
        model_cfg = dict(config.get("model", {}))

        # Enforce explicit dimensions or fail
        if model_type == "fnn":
            hidden_dim = config.get("hidden_dim")
            if hidden_dim is None:
                raise ValueError("Configuration Error: 'hidden_dim' must be specified for FNN.")

            model_cfg["layer_dims"] = [
                int(input_shape[-1]),
                int(hidden_dim),
                int(meta_n_outputs)
            ]
        else:
            # RNN types
            hidden_dim = config.get("hidden_dim")
            if hidden_dim is None:
                raise ValueError(f"Configuration Error: 'hidden_dim' must be specified for {model_type}.")

            model_cfg.setdefault("input_dim", int(input_shape[-1]))
            model_cfg["hidden_dim"] = int(hidden_dim)
            model_cfg.setdefault("output_dim", int(meta_n_outputs))
            model_cfg.setdefault("return_sequences", False)
            model_cfg.setdefault("return_hidden", False)

        factory_cfg = {"type": model_type, "model": model_cfg, "training": training_cfg}
        model = FlaxModelFactory.create_model(factory_cfg)

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

        # Readout Config
        readout_cfg = config.get("readout", {})
        readout = RidgeRegression(
            alpha=float(readout_cfg.get("alpha", 1e-3)),
            use_intercept=bool(readout_cfg.get("fit_intercept", True)),
        )

        readout_mode = reservoir_params.get("state_aggregation")
        if not readout_mode:
            raise ValueError(
                "Configuration Error: 'state_aggregation' is missing in reservoir_params. "
                "Define it in the preset or override explicitly."
            )

        model = ReservoirModel(reservoir=node, readout=readout, preprocess=preprocess, readout_mode=readout_mode)

        # --- Architecture Summary ---
        raw_input_dim = int(input_shape[-1])
        res_units = int(node.n_units)
        out_dim = int(meta_n_outputs)

        print("=" * 40)
        print(f"🏗️  Model Architecture: {model_type.upper()}")
        print("=" * 40)
        print(f"1. Input Data      : {raw_input_dim} features")
        if preprocess:
            print(f"2. Preprocessing   : Expanded to {effective_input_dim} (PolyDegree={degree}, Bias={include_bias})")
        else:
            print(f"2. Preprocessing   : None (Pass-through)")
        print(f"3. Reservoir       : {effective_input_dim} -> {res_units} units (Recurrent)")
        print(f"4. Readout         : {res_units} -> {out_dim} outputs")
        print("-" * 40)
        print(f"🔗 Topology String : {raw_input_dim} -> {effective_input_dim} -> {res_units} -> {out_dim}")
        print("=" * 40)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # --- 3a. Validation Split for ridge search ---
    val_tuple = None
    val_size = float(training_cfg.get("val_size", 0.0)) if training_cfg else 0.0
    if val_size > 0.0 and len(train_X) > 1:
        val_count = max(1, int(len(train_X) * val_size))
        train_count = len(train_X) - val_count
        if train_count < 1:
            train_count = len(train_X) - 1
            val_count = 1
        val_X = train_X[train_count:]
        val_y = train_y[train_count:]
        train_X = train_X[:train_count]
        train_y = train_y[:train_count]
        val_tuple = (val_X, val_y)

    # --- 3. Execution ---
    metric = "accuracy" if is_classification else "mse"
    runner = UniversalPipeline(model, config.get("save_path"), metric=metric)
    results = runner.run(
        train_X,
        train_y,
        test_X,
        test_y,
        validation=val_tuple,
        training_cfg=training_cfg,
    )

    # Log ridge search results if available
    train_res = results.get("train", {}) if isinstance(results, dict) else {}
    if isinstance(train_res, dict) and "search_history" in train_res:
        history = train_res.get("search_history", {})
        best_lam = train_res.get("best_lambda")
        metric_name = train_res.get("metric", "score")

        print("\n" + "=" * 40)
        print(f"🔎 Ridge Hyperparameter Search ({metric_name})")
        print("-" * 40)
        sorted_lambdas = sorted(history.keys())
        for lam in sorted_lambdas:
            score = history[lam]
            marker = " 🏆 Best" if (best_lam is not None and jnp.isclose(lam, best_lam)) else ""
            print(f"   λ = {float(lam):.2e} : Val Score = {float(score):.4f}{marker}")
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

            train_pred_np = np.asarray(model.predict(train_X))
            test_pred_np = np.asarray(model.predict(test_X))
            if train_pred_np.ndim > 1:
                train_pred_np = np.argmax(train_pred_np, axis=-1)
            if test_pred_np.ndim > 1:
                test_pred_np = np.argmax(test_pred_np, axis=-1)

            filename = f"outputs/{dataset_name}_{model_type}_raw_nr{reservoir_params["n_units"]}_confusion.png"
            plot_classification_results(
                train_labels=train_labels_np,
                test_labels=test_labels_np,
                train_predictions=train_pred_np,
                test_predictions=test_pred_np,
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
