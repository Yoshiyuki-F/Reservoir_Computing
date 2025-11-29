"""/home/yoshi/PycharmProjects/Reservoir/pipelines/run.py
Unified Pipeline Runner for JAX-based Models and Datasets."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import jax.numpy as jnp

# Core Imports
from reservoir.models import FlaxModelFactory
from reservoir.models.reservoir.factory import ReservoirFactory
from pipelines.generic_runner import UniversalPipeline
from reservoir.data.registry import DatasetRegistry

# Ensure dataset loaders are registered
from reservoir.data import loaders as _data_loaders  # noqa: F401

_DATASET_PRESETS = Path(__file__).resolve().parents[1] / "presets" / "datasets" / "datasets.json"


@lru_cache()
def _load_dataset_presets() -> Dict[str, Any]:
    if _DATASET_PRESETS.exists():
        return json.loads(_DATASET_PRESETS.read_text())
    return {}


def _dataset_meta(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    name = str(config.get("dataset", "sine_wave")).lower()
    presets = _load_dataset_presets()
    return name, presets.get(name, {})


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
    dataset = str(config.get("dataset", "sine_wave"))
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

    dataset_name, dataset_meta = _dataset_meta(config)
    preset_type = str(dataset_meta.get("type", "")).lower()
    override_cls = config.get("is_classification")
    if override_cls is not None:
        is_classification = bool(override_cls)
    elif preset_type in {"classification", "regression"}:
        is_classification = preset_type == "classification"
    else:
        raise ValueError(f"Unknown preset type: {preset_type}")
    meta_n_outputs = dataset_meta.get("n_output")
    
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
        raise ValueError(
            "Classification tasks require 'n_output' to be specified in dataset presets."
        )
    default_output_dim = int(meta_n_outputs)

    # 2. Model Creation
    input_shape = train_X.shape[1:]
    print(f"Initializing {model_type.upper()} model via Factory...")
    
    if model_type in ["fnn", "rnn", "lstm", "gru"]:
        # FlaxModelFactory expects a dict with type/model/training keys
        model_cfg = dict(config.get("model", {}))
        training_cfg = dict(config.get("training", {}))
        if is_classification:
            training_cfg["classification"] = True
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
    elif model_type in ["reservoir", "esn", "classical", "quantum_analog", "quantum_gate"]:
        reservoir_type = config.get("reservoir_type") or ("classical" if model_type == "reservoir" else model_type)
        reservoir_cfg = config.get("reservoir_config", config)
        if isinstance(reservoir_cfg, dict):
            params = reservoir_cfg.setdefault("params", {})
            params.setdefault("n_inputs", int(input_shape[-1]))
            params.setdefault("n_hidden_layer", int(config.get("hidden_dim", params.get("n_hidden_layer", 100))))
            if "n_outputs" not in params:
                params["n_outputs"] = int(default_output_dim)
        # Reservoir auto-detects classification from data, no guard needed.
        model = ReservoirFactory.create(
            reservoir_cfg,
            input_shape=input_shape,
            reservoir_type=reservoir_type,
            backend=config.get("backend"),
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # 3. Execution
    runner = UniversalPipeline(model, config.get("save_path"))
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
