"""Dynamic experiment orchestration utilities for any model type."""

import json
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np

from configs.core import ExperimentConfig, ConfigComposer
from pipelines.metrics import calculate_mse, calculate_mae
from pipelines.preprocessing import denormalize_data
from pipelines.plotting import plot_prediction_results


def get_model_factory(model_type: str):
    """Get the appropriate model factory based on model type."""
    if "reservoir" in model_type or "quantum" in model_type:
        from models.reservoir import ReservoirComputerFactory
        return ReservoirComputerFactory
    elif "ffn" in model_type:
        # Future: from models.ffn import FFNFactory
        raise NotImplementedError("FFN models not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_data_preparation_function(model_type: str):
    """Get the appropriate data preparation function based on model type."""
    # Data preparation is now model-agnostic and located in pipelines
    from pipelines import prepare_experiment_data
    return prepare_experiment_data


def _save_config_snapshot(
    config: ExperimentConfig,
    output_filename: str,
    metrics: Dict[str, Any],
    ridge_log: Optional[list],
) -> None:
    snapshot = config.model_dump(mode='json')
    snapshot.setdefault('results', {})
    snapshot['results']['metrics'] = metrics
    if ridge_log is not None:
        snapshot['results']['ridge_search'] = ridge_log

    snapshot_path = Path('outputs') / f"{Path(output_filename).stem}_config.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open('w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2)
    print(f"📝 Saved config snapshot -> {snapshot_path}")


def run_dynamic_experiment(
    dataset_name: str,
    model_name: str,
    training_name: str = "standard",
    show_training: bool = False,
    backend: Optional[str] = None,
    force_cpu: bool = False
) -> Tuple[Optional[float], float, Optional[float], float]:
    """Run an experiment with dynamic configuration.

    Args:
        dataset_name: Name of dataset config (e.g., 'sine_wave', 'lorenz')
        model_name: Name of model config (e.g., 'reservoir_standard', 'quantum_standard')
        training_name: Name of training config (default: 'standard')
        show_training: Whether to show training data in visualization
        backend: Compute backend ('cpu' or 'gpu')
        force_cpu: Force CPU usage

    Returns:
        Tuple of (train_mse, test_mse, train_mae, test_mae)
    """

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

    # Compose configuration
    composer = ConfigComposer()
    composed_config = composer.compose_experiment(dynamic_experiment)
    legacy_config = composer.compose_legacy_format(composed_config)

    # Create experiment config object
    experiment_config = ExperimentConfig(**legacy_config)

    # Determine model type and quantum mode
    is_quantum = "quantum" in model_name
    model_type = model_name

    # Get data preparation function and prepare data
    prepare_data_fn = get_data_preparation_function(model_type)
    dataset = prepare_data_fn(experiment_config, quantum_mode=is_quantum)

    # Run the experiment
    return run_experiment(
        experiment_config,
        dataset,
        backend=backend,
        quantum_mode=is_quantum,
        model_type=model_type
    )


def run_experiment(
    demo_config: ExperimentConfig,
    dataset: Any,  # ExperimentDataset type
    backend: Optional[str] = None,
    quantum_mode: bool = False,
    model_type: Optional[str] = None,
) -> Tuple[Optional[float], float, Optional[float], float]:
    """Execute training, prediction, and reporting for a prepared dataset."""

    print(f"=== {demo_config.demo.title} ===")

    # Determine model type if not provided
    if model_type is None:
        if quantum_mode:
            model_type = "quantum"
        else:
            model_type = "reservoir"

    # Get the appropriate factory for this model type
    ModelFactory = get_model_factory(model_type)

    if quantum_mode or "quantum" in model_type:
        if demo_config.quantum_reservoir is None:
            raise ValueError("Quantum mode enabled but quantum_reservoir config is missing")
        rc = ModelFactory.create_reservoir(
            'quantum', demo_config.quantum_reservoir, backend
        )
    else:
        if demo_config.reservoir is None:
            raise ValueError("Classical mode requires reservoir config")
        rc = ModelFactory.create_reservoir(
            'classical', demo_config.reservoir, backend
        )

    reservoir_info = rc.get_reservoir_info()
    print(f"Reservoir情報: {reservoir_info}")

    if quantum_mode or "quantum" in model_type:
        n_qubits = reservoir_info.get("n_qubits")
        measurement_basis = reservoir_info.get("measurement_basis", "pauli-z")
        if measurement_basis == "multi-pauli" and n_qubits is not None:
            from math import comb

            feature_dim = 3 * int(n_qubits) + comb(int(n_qubits), 2)
            print(f"🧮 Quantum feature dimension: {feature_dim} (3×{n_qubits} one-body + {comb(int(n_qubits),2)} two-body)")

    print("訓練中...")
    lambda_candidates = list(demo_config.training.ridge_lambdas or [])
    if not lambda_candidates:
        lambda_candidates = [1e-6, 1e-5, 1e-4, 1e-3]

    rc.train(
        dataset.train_input,
        dataset.train_target,
        ridge_lambdas=lambda_candidates,
    )

    if getattr(rc, "ridge_search_log", None):
        print("🔍 Ridge λ grid search")
        for entry in rc.ridge_search_log:
            lam = entry["lambda"]
            mse = entry["train_mse"]
            print(f"  λ={lam:.2e} -> train MSE={mse:.6f}")
        if rc.best_ridge_lambda is not None:
            print(f"✅ Selected λ={rc.best_ridge_lambda:.2e}")

    print("予測中...")
    test_predictions_norm = rc.predict(dataset.test_input)
    test_target_norm = dataset.test_target

    test_predictions = denormalize_data(
        test_predictions_norm, dataset.target_mean, dataset.target_std
    )
    test_target_orig = denormalize_data(
        test_target_norm, dataset.target_mean, dataset.target_std
    )

    test_mse = calculate_mse(test_predictions, test_target_orig)
    test_mae = calculate_mae(test_predictions, test_target_orig)

    train_mse = train_mae = None
    train_predictions_norm = train_target_norm = None
    train_predictions_orig = train_target_orig = None

    if dataset.train_input.size > 0:
        train_predictions_norm = rc.predict(dataset.train_input)
        train_target_norm = dataset.train_target

        train_predictions_orig = denormalize_data(
            train_predictions_norm, dataset.target_mean, dataset.target_std
        )
        train_target_orig = denormalize_data(
            train_target_norm, dataset.target_mean, dataset.target_std
        )

        train_mse = calculate_mse(train_predictions_orig, train_target_orig)
        train_mae = calculate_mae(train_predictions_orig, train_target_orig)

        if demo_config.demo.show_training:
            print(f"訓練 MSE: {train_mse:.6f}, MAE: {train_mae:.6f}")

    print(f"テスト MSE: {test_mse:.6f}, MAE: {test_mae:.6f}")

    output_filename = demo_config.demo.filename

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

    if getattr(rc, "best_ridge_lambda", None) is not None:
        best_lambda = float(rc.best_ridge_lambda)
        metrics_caption["Ridge λ"] = f"{best_lambda:.2e}"
        metrics_snapshot["best_ridge_lambda"] = best_lambda

    ridge_log = getattr(rc, "ridge_search_log", None)

    plot_prediction_results(
        test_target_orig,
        test_predictions,
        time_indices,
        demo_config.demo.title,
        output_filename,
        train_target_orig,
        train_predictions_orig,
        dataset.train_size if dataset.train_input.size > 0 else None,
        y_axis_label=demo_config.demo.y_axis_label,
        metrics_info=metrics_caption,
        add_test_zoom=demo_config.demo.add_test_zoom,
        zoom_range=demo_config.demo.zoom_range,
    )

    _save_config_snapshot(demo_config, output_filename, metrics_snapshot, ridge_log)

    return train_mse, test_mse, train_mae, test_mae


def run_experiment_from_config(
    config_path: str,
    backend: Optional[str] = None,
    quantum_mode: bool = False,
) -> Tuple[Optional[float], float, Optional[float], float]:
    """High-level helper that loads config, prepares data, and runs the experiment."""

    demo_config = ExperimentConfig.from_json(config_path)
    dataset = prepare_experiment_data(demo_config, quantum_mode=quantum_mode)
    return run_experiment(demo_config, dataset, backend=backend, quantum_mode=quantum_mode)
