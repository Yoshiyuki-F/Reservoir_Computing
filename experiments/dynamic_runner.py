"""Dynamic experiment orchestration utilities for any model type."""

import json
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from functools import lru_cache

import numpy as np
import jax.numpy as jnp

from configs.core import ExperimentConfig, ConfigComposer
from pipelines.metrics import calculate_mse, calculate_mae
from pipelines.preprocessing import denormalize_data
from pipelines.plotting import plot_prediction_results, plot_classification_results
from pipelines.data_preparation import ExperimentDatasetClassification


@lru_cache()
def _load_config_json(relative_path: str) -> Dict[str, Any]:
    path = Path(relative_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / relative_path
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


@lru_cache()
def _load_basic_reservoir_config() -> Dict[str, Any]:
    return _load_config_json('configs/models/basic_reservoir.json')


@lru_cache()
def _load_quantum_standard_config() -> Dict[str, Any]:
    return _load_config_json('configs/models/quantum_standard.json')


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

    snapshot_path = Path('outputs') / f"{Path(output_filename).stem}_config.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open('w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, default=_json_default)
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
        model_name: Name of model config (e.g., 'classic_standard', 'quantum_standard')
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

    if dataset_name == 'mnist':
        training_cfg = legacy_config.get("training", {}).copy()
        training_cfg["task_type"] = "classification"
        legacy_config["training"] = training_cfg

    # Create experiment config object
    experiment_config = ExperimentConfig(**legacy_config)

    # Determine model type and quantum mode
    is_quantum = "quantum" in model_name or experiment_config.model.model_type == "quantum"
    model_type = experiment_config.model.model_type or model_name

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

    if isinstance(dataset, ExperimentDatasetClassification):
        input_dim = int(dataset.train_sequences.shape[-1])
        combined_labels = jnp.concatenate(
            [dataset.train_labels, dataset.test_labels]
        )
        if combined_labels.size > 0:
            num_classes = int(jnp.max(combined_labels).item()) + 1
        else:
            num_classes = 1

        if demo_config.reservoir is None:
            demo_config.reservoir = {}
        demo_config.reservoir['n_inputs'] = input_dim
        demo_config.reservoir.setdefault('n_outputs', num_classes)
        demo_config.reservoir.setdefault('state_aggregation', 'mean')

    if quantum_mode or "quantum" in model_type:
        if demo_config.quantum_reservoir is None:
            raise ValueError("Quantum mode enabled but quantum_reservoir config is missing")
        quantum_base = _load_quantum_standard_config().get('params', {})
        basic_base = _load_basic_reservoir_config()
        config_sequence = [
            {'params': basic_base},
            {'params': quantum_base},
            demo_config.quantum_reservoir,
        ]
        rc = ModelFactory.create_reservoir(
            'quantum', config_sequence, backend
        )
    else:
        if demo_config.reservoir is None:
            raise ValueError("Classical mode requires reservoir config")
        basic_base = _load_basic_reservoir_config()
        config_sequence = [
            {'params': basic_base},
            demo_config.reservoir,
        ]
        rc = ModelFactory.create_reservoir(
            'classical', config_sequence, backend
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

    lambda_candidates = list(demo_config.training.ridge_lambdas or [])
    if not lambda_candidates:
        lambda_candidates = [1e-6, 1e-5, 1e-4, 1e-3]

    if isinstance(dataset, ExperimentDatasetClassification):
        if quantum_mode or "quantum" in model_type:
            raise NotImplementedError("Quantum classification mode is not yet supported")

        print("訓練中 (classification)...")
        rc.train_classification(
            dataset.train_sequences,
            dataset.train_labels,
            ridge_lambdas=lambda_candidates,
            num_classes=10,
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
        train_logits = rc.predict_classification(dataset.train_sequences)
        test_logits = rc.predict_classification(dataset.test_sequences)

        train_one_hot = jnp.zeros((dataset.train_labels.shape[0], 10), dtype=jnp.float64)
        train_one_hot = train_one_hot.at[jnp.arange(dataset.train_labels.shape[0]), dataset.train_labels].set(1.0)
        test_one_hot = jnp.zeros((dataset.test_labels.shape[0], 10), dtype=jnp.float64)
        test_one_hot = test_one_hot.at[jnp.arange(dataset.test_labels.shape[0]), dataset.test_labels].set(1.0)

        train_mse = float(calculate_mse(train_logits, train_one_hot))
        test_mse = float(calculate_mse(test_logits, test_one_hot))

        train_pred = jnp.argmax(train_logits, axis=1)
        test_pred = jnp.argmax(test_logits, axis=1)
        train_accuracy = float(jnp.mean(train_pred == dataset.train_labels))
        test_accuracy = float(jnp.mean(test_pred == dataset.test_labels))

        print(f"Accuracy: train={train_accuracy:.4f}, test={test_accuracy:.4f}")

        metrics_caption = {
            "Train MSE": train_mse,
            "Test MSE": test_mse,
            "Train Acc": f"{train_accuracy:.4f}",
            "Test Acc": f"{test_accuracy:.4f}",
        }
        metrics_snapshot = {
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }

        if getattr(rc, "best_ridge_lambda", None) is not None:
            best_lambda = float(rc.best_ridge_lambda)
            metrics_caption["Ridge λ"] = f"{best_lambda:.2e}"
            metrics_snapshot["best_ridge_lambda"] = best_lambda

        ridge_log = getattr(rc, "ridge_search_log", None)

        # 可視化 (classification)
        labels_arrays = [
            np.asarray(dataset.train_labels),
            np.asarray(dataset.test_labels),
            np.asarray(train_pred),
            np.asarray(test_pred),
        ]
        detected_classes = max(
            (int(arr.max()) if arr.size > 0 else -1) for arr in labels_arrays
        )
        class_count = max(detected_classes + 1, 10)
        class_names = [str(i) for i in range(class_count)]
        plot_classification_results(
            dataset.train_labels,
            dataset.test_labels,
            train_pred,
            test_pred,
            demo_config.demo.title,
            demo_config.demo.filename,
            metrics_info=metrics_caption,
            class_names=class_names,
        )

        extra_results = {
            "classification": {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "reservoir_info": reservoir_info,
            }
        }
        _save_config_snapshot(
            demo_config,
            demo_config.demo.filename,
            metrics_snapshot,
            ridge_log,
            extra=extra_results,
        )

        return train_mse, test_mse, train_accuracy, test_accuracy

    print("訓練中...")
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
