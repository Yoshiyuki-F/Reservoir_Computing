"""Dynamic experiment orchestration utilities for any model type."""

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

    print(f"Reservoir情報: {rc.get_reservoir_info()}")

    print("訓練中...")
    rc.train(dataset.train_input, dataset.train_target, reg_param=demo_config.training.reg_param)

    print("予測中...")
    test_predictions = rc.predict(dataset.test_input)

    test_predictions = denormalize_data(test_predictions, dataset.target_mean, dataset.target_std)
    test_target_orig = denormalize_data(dataset.test_target, dataset.target_mean, dataset.target_std)

    test_mse = calculate_mse(test_predictions, test_target_orig)
    test_mae = calculate_mae(test_predictions, test_target_orig)

    train_mse = train_mae = None
    train_predictions_orig = train_target_orig = None

    if demo_config.demo.show_training:
        train_predictions = rc.predict(dataset.train_input)
        train_predictions_orig = denormalize_data(train_predictions, dataset.target_mean, dataset.target_std)
        train_target_orig = denormalize_data(dataset.train_target, dataset.target_mean, dataset.target_std)

        train_mse = calculate_mse(train_predictions_orig, train_target_orig)
        train_mae = calculate_mae(train_predictions_orig, train_target_orig)

        print(f"訓練 MSE: {train_mse:.6f}, MAE: {train_mae:.6f}")

    print(f"テスト MSE: {test_mse:.6f}, MAE: {test_mae:.6f}")

    output_filename = demo_config.demo.filename
    if quantum_mode:
        filename_path = Path(output_filename)
        suffix = filename_path.suffix or ".png"
        output_filename = f"{filename_path.stem}_quantum{suffix}"

    time_indices = np.arange(test_target_orig.shape[0])

    if demo_config.demo.show_training:
        plot_prediction_results(
            test_target_orig,
            test_predictions,
            time_indices,
            demo_config.demo.title,
            output_filename,
            train_target_orig,
            train_predictions_orig,
            dataset.train_size,
        )
    else:
        plot_prediction_results(
            test_target_orig,
            test_predictions,
            time_indices,
            demo_config.demo.title,
            output_filename,
        )

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
