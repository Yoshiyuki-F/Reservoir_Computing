"""/home/yoshi/PycharmProjects/Reservoir/pipelines/run.py
Unified Pipeline Runner for JAX-based Models and Datasets."""

from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp

# Core Imports
from reservoir.models import FlaxModelFactory
from pipelines.generic_runner import UniversalPipeline
from reservoir.components import FeatureScaler, DesignMatrix, RidgeRegression, TransformerSequence
from reservoir.data.presets import DATASET_REGISTRY, DatasetPreset
from reservoir.data.registry import DatasetRegistry
from reservoir.models.orchestrator import ReservoirModel
from reservoir.models.presets import MODEL_REGISTRY, get_model_preset
from reservoir.training.presets import TRAINING_REGISTRY, get_training_preset
from reservoir.models.reservoir.classical import ClassicalReservoir

# Ensure dataset loaders are registered
from reservoir.data import loaders as _data_loaders  # noqa: F401


def _dataset_meta(config: Dict[str, Any]) -> Tuple[str, Optional[DatasetPreset]]:
    name = DATASET_REGISTRY.normalize_name(config.get("dataset", "sine_wave"))
    return name, DATASET_REGISTRY.get(name)


def _resolve_training_config(config: Dict[str, Any], is_classification: bool) -> Dict[str, Any]:
    preset_name = config.get("training_preset", "standard")
    preset: TrainingConfig = get_training_preset(preset_name)
    merged = preset.to_dict()
    overrides = dict(config.get("training", {}) or {})
    merged.update(overrides)
    if is_classification:
        merged["classification"] = True
    return merged


def _resolve_reservoir_params(config: Dict[str, Any]) -> Dict[str, Any]:
    preset_name = config.get("reservoir_preset") or config.get("reservoir_type") or "classical"
    preset: ModelPreset = get_model_preset(preset_name)
    base_params = preset.to_params()
    overrides = dict(config.get("reservoir", {}) or {})
    top_level_keys = [
        "n_units",
        "hidden_dim",
        "input_scale",
        "input_scaling",
        "spectral_radius",
        "leak_rate",
        "alpha",
        "connectivity",
        "sparsity",
        "noise_rc",
        "bias_scale",
        "random_seed",
        "seed",
        "use_design_matrix",
        "poly_degree",
        "poly_bias",
    ]
    for key in top_level_keys:
        if key in config:
            overrides.setdefault(key, config[key])

    merged = {**base_params, **overrides}
    if "hidden_dim" in merged:
        merged["n_units"] = int(merged["hidden_dim"])
    merged["n_units"] = int(merged.get("n_units", 100))
    if "input_scaling" in merged and "input_scale" not in merged:
        merged["input_scale"] = merged["input_scaling"]
    if "sparsity" in merged and "connectivity" not in merged:
        merged["connectivity"] = merged["sparsity"]
    if "random_seed" in merged and "seed" not in merged:
        merged["seed"] = merged["random_seed"]
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
    dataset = DATASET_REGISTRY.normalize_name(config.get("dataset", "sine_wave"))
    loader = DatasetRegistry.get(dataset)
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
    The Unified Entry Point.
    Orchestrates data loading, model creation, and execution.
    """
    # 1. Data Preparation
    if train_X is None or train_y is None:
        print(f"Loading dataset: {config.get('dataset', 'sine_wave')}...")
        X, y = _load_dataset(config)
        split_idx = int(0.8 * len(X))
        train_X, test_X = X[:split_idx], X[split_idx:]
        train_y, test_y = y[:split_idx], y[split_idx:]

    dataset_name, dataset_preset = _dataset_meta(config)
    if dataset_name in {"mnist", "fashion_mnist"}:
        num_classes = int(dataset_preset.config.n_output if dataset_preset else 10)
        print(f"Converting labels to one-hot vectors (classes={num_classes})...")
        train_y = jax.nn.one_hot(train_y.astype(int), num_classes)
        test_y = jax.nn.one_hot(test_y.astype(int), num_classes)
        print("Normalizing image data to [0, 1] range (div by 255)...")
        train_X = train_X.astype(jnp.float32) / 255.0
        if test_X is not None:
            test_X = test_X.astype(jnp.float32) / 255.0
        config["use_preprocessing"] = False

    preset_type = dataset_preset.task_type.lower() if dataset_preset else "regression"
    override_cls = config.get("is_classification")
    if override_cls is not None:
        is_classification = bool(override_cls)
    elif preset_type in {"classification", "regression"}:
        is_classification = preset_type == "classification"
    else:
        raise ValueError(f"Unknown preset type: {preset_type}")
    meta_n_outputs = dataset_preset.config.n_output if dataset_preset else None
    
    # --- Shape Adjustment Logic ---
    model_type = config.get("model_type").lower()

    # FNN expects flattened input: (N, Features)
    if model_type == "fnn":
        if train_X.ndim > 2:
            print(f"Flattening input for FNN: {train_X.shape} -> (N, Flattened)")
            train_X = train_X.reshape(train_X.shape[0], -1)
            if test_X is not None:
                test_X = test_X.reshape(test_X.shape[0], -1)

    # RNN/Reservoir expects sequence input: (N, Time, Features)
    elif model_type in ["rnn", "reservoir", "esn", "classical"]:
        if train_X.ndim != 3:
            raise ValueError(
                f"Model type '{model_type}' requires 3D input (Batch, Time, Features). "
                f"Got shape {train_X.shape}. Please reshape your data source."
            )

    print(f"Data Shapes -> Train: {train_X.shape}, Test: {test_X.shape if test_X is not None else 'None'}")

    # Determine default output dimension from presets/data
    if meta_n_outputs is None:
        print("Dataset metadata missing 'n_output'; defaulting to regression with inferred output dim 1.")
        meta_n_outputs = 1
        is_classification = False
    default_output_dim = int(meta_n_outputs)

    # 2. Model Creation
    input_shape = train_X.shape[1:]
    print(f"Initializing {model_type.upper()} model via Factory...")
    
    if model_type in ["fnn", "rnn", "lstm", "gru"]:
        # FlaxModelFactory expects a dict with type/model/training keys
        model_cfg = dict(config.get("model", {}))
        training_cfg = _resolve_training_config(config, is_classification)
        # Inject shapes
        if model_type == "fnn":
            model_cfg.setdefault(
                "layer_dims",
                [int(input_shape[-1]), int(config.get("hidden_dim", 128)), int(default_output_dim)],
            )
        else:
            model_cfg.setdefault("input_dim", int(input_shape[-1]))
            model_cfg.setdefault("hidden_dim", int(config.get("hidden_dim", 64)))
            model_cfg.setdefault("output_dim", int(default_output_dim))
            model_cfg.setdefault("return_sequences", False)
            model_cfg.setdefault("return_hidden", False)
        factory_cfg = {"type": model_type, "model": model_cfg, "training": training_cfg}
        model = FlaxModelFactory.create_model(factory_cfg)
    elif model_type in ["reservoir", "esn", "classical"]:
        reservoir_params = _resolve_reservoir_params(config)
        reservoir_preset = config.get("reservoir_preset") or config.get("reservoir_type") or "classical"
        preset_obj = get_model_preset(reservoir_preset)

        preprocess_steps = []
        effective_input_dim = int(input_shape[-1])
        if dataset_name not in {"mnist", "fashion_mnist"} and config.get("use_preprocessing", True):
            preprocess_steps.append(FeatureScaler())

        use_dm = bool(reservoir_params.get("use_design_matrix", config.get("use_design_matrix", False)))
        degree = int(reservoir_params.get("poly_degree", config.get("poly_degree", 2)))
        include_bias = bool(reservoir_params.get("poly_bias", config.get("poly_bias", False)))
        if use_dm:
            print(f"Adding DesignMatrix (degree={degree}, bias={include_bias})...")
            preprocess_steps.append(DesignMatrix(degree=degree, include_bias=include_bias))
            factor = degree if degree > 0 else 1
            effective_input_dim = effective_input_dim * factor + (1 if include_bias else 0)
        preprocess = TransformerSequence(preprocess_steps) if preprocess_steps else None
        readout_mode = "flatten" if is_classification else "auto"

        n_units = int(reservoir_params.get("n_units", config.get("hidden_dim", 100)))
        input_scale = float(reservoir_params.get("input_scale", 0.6))
        spectral_radius = float(reservoir_params.get("spectral_radius", 1.3))
        leak_rate = float(reservoir_params.get("leak_rate", reservoir_params.get("alpha", 0.2)))
        connectivity = float(reservoir_params.get("connectivity", 0.1))
        noise_rc = float(reservoir_params.get("noise_rc", reservoir_params.get("noise_level", 0.0)))
        bias_scale = float(reservoir_params.get("bias_scale", reservoir_params.get("input_bias", 1.0)))
        seed = int(reservoir_params.get("seed", config.get("random_seed", 42)))

        print(f"Using reservoir preset '{preset_obj.name}' with {n_units} units.")

        node = ClassicalReservoir(
            n_inputs=effective_input_dim,
            n_units=n_units,
            input_scale=input_scale,
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
            connectivity=connectivity,
            noise_rc=noise_rc,
            bias_scale=bias_scale,
            seed=seed,
        )
        readout_cfg = config.get("readout", {})
        readout = RidgeRegression(
            alpha=float(readout_cfg.get("alpha", 1e-3)),
            use_intercept=bool(readout_cfg.get("fit_intercept", True)),
        )
        model = ReservoirModel(reservoir=node, readout=readout, preprocess=preprocess, readout_mode=readout_mode)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # 3. Execution
    metric = "accuracy" if dataset_name in {"mnist", "fashion_mnist"} else "mse"
    runner = UniversalPipeline(model, config.get("save_path"), metric=metric)
    results = runner.run(train_X, train_y, test_X, test_y)

    # 4. Persistence
    if save_path and hasattr(model, 'save'):
        print(f"Saving model to {save_path}...")
        model.save(save_path)

    return results

# --- Convenience Wrappers ---

def run_fnn_pipeline(config: Dict[str, Any], save_path: Optional[str] = None) -> Dict[str, Any]:
    config["model_type"] = "fnn"
    return run_pipeline(config, save_path=save_path)

def run_rnn_pipeline(config: Dict[str, Any], save_path: Optional[str] = None) -> Dict[str, Any]:
    config["model_type"] = "rnn"
    return run_pipeline(config, save_path=save_path)

def run_reservoir_pipeline(config: Dict[str, Any], save_path: Optional[str] = None) -> Dict[str, Any]:
    if "reservoir_type" not in config:
        config["reservoir_type"] = "classical"
    config["model_type"] = "reservoir"
    return run_pipeline(config, save_path=save_path)
