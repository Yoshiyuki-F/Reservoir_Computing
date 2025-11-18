"""Dynamic experiment orchestration utilities for any model type."""

import json
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING, cast, Callable, Sequence
from pathlib import Path
from functools import lru_cache

import numpy as np
import jax.numpy as jnp

from core_lib.core import ExperimentConfig, ConfigComposer
from pipelines import prepare_experiment_data
from core_lib.utils import (
    calculate_mse,
    calculate_mae,
    denormalize_data,
    plot_prediction_results,
    plot_classification_results,
)
from pipelines.data_preparation import ExperimentDatasetClassification
from pipelines.classical_reservoir_pipeline import (
    train_reservoir_regression,
    predict_reservoir_regression,
    train_reservoir_classification,
    predict_reservoir_classification,
)
from pipelines.gatebased_quantum_pipeline import (
    train_quantum_reservoir_regression,
    predict_quantum_reservoir_regression,
    train_quantum_reservoir_classification,
    predict_quantum_reservoir_classification,
)

if TYPE_CHECKING:
    from core_lib.models.reservoir.classical import ReservoirComputer


@lru_cache()
def _load_config_json(relative_path: str) -> Dict[str, Any]:
    path = Path(relative_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / relative_path
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


@lru_cache()
def _load_shared_reservoir_config() -> Dict[str, Any]:
    return _load_config_json('configs/models/shared_reservoir_params.json')


@lru_cache()
def _load_gatebased_quantum_config() -> Dict[str, Any]:
    return _load_config_json('configs/models/gatebased_quantum.json')


@lru_cache()
def _load_analog_quantum_config() -> Dict[str, Any]:
    return _load_config_json('configs/models/analog_quantum.json')


def get_model_factory(model_type: str):
    """Get the appropriate model factory based on model type."""
    if "reservoir" in model_type or "quantum" in model_type:
        from core_lib.models.reservoir import ReservoirComputerFactory
        return ReservoirComputerFactory
    elif "ffn" in model_type:
        # Future: from models.ffn import FFNFactory
        raise NotImplementedError("FFN models not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


TrainFn = Callable[[Any, jnp.ndarray, jnp.ndarray, Optional[Sequence[float]]], None]
PredictFn = Callable[[Any, jnp.ndarray], jnp.ndarray]

ClassTrainFn = Callable[
    [Any, jnp.ndarray, jnp.ndarray, Optional[Sequence[float]], int, bool],
    Optional[jnp.ndarray],
]
ClassPredictFn = Callable[
    [Any, jnp.ndarray, Optional[jnp.ndarray], str],
    jnp.ndarray,
]


def _legacy_train_regression(
    rc: Any,
    input_data: jnp.ndarray,
    target_data: jnp.ndarray,
    ridge_lambdas: Optional[Sequence[float]],
) -> None:
    """Legacy training path for models that still own their train() logic."""
    rc.train(input_data, target_data, ridge_lambdas=ridge_lambdas)


def _legacy_predict_regression(
    rc: Any,
    input_data: jnp.ndarray,
) -> jnp.ndarray:
    """Legacy prediction path for models that still own their predict() logic."""
    return rc.predict(input_data)


REGRESSION_PIPELINES: Dict[str, Tuple[TrainFn, PredictFn]] = {
    "classical": (train_reservoir_regression, predict_reservoir_regression),
    "gatebased_quantum": (train_quantum_reservoir_regression, predict_quantum_reservoir_regression),
    "analog_quantum_legacy": (_legacy_train_regression, _legacy_predict_regression),
}


def _train_classical_classification(
    rc: Any,
    sequences: jnp.ndarray,
    labels: jnp.ndarray,
    ridge_lambdas: Optional[Sequence[float]],
    num_classes: int,
    return_features: bool,
) -> Optional[jnp.ndarray]:
    return train_reservoir_classification(
        rc,
        sequences,
        labels,
        ridge_lambdas=ridge_lambdas,
        num_classes=num_classes,
        return_features=return_features,
    )


def _predict_classical_classification(
    rc: Any,
    sequences: jnp.ndarray,
    cached_features: Optional[jnp.ndarray],
    desc: str,
) -> jnp.ndarray:
    if cached_features is not None:
        return predict_reservoir_classification(
            rc,
            sequences=None,
            precomputed_features=cached_features,
            progress_desc=desc,
        )
    return predict_reservoir_classification(
        rc,
        sequences=sequences,
        precomputed_features=None,
        progress_desc=desc,
    )


def _train_quantum_classification(
    rc: Any,
    sequences: jnp.ndarray,
    labels: jnp.ndarray,
    ridge_lambdas: Optional[Sequence[float]],
    num_classes: int,
    return_features: bool,
) -> Optional[jnp.ndarray]:
    return train_quantum_reservoir_classification(
        rc,
        sequences,
        labels,
        ridge_lambdas=ridge_lambdas,
        num_classes=num_classes,
        return_features=return_features,
    )


def _predict_quantum_classification(
    rc: Any,
    sequences: jnp.ndarray,
    cached_features: Optional[jnp.ndarray],
    desc: str,
) -> jnp.ndarray:
    if cached_features is not None:
        return predict_quantum_reservoir_classification(
            rc,
            sequences=None,
            progress_desc=desc,
            progress_position=0,
            precomputed_features=cached_features,
        )
    return predict_quantum_reservoir_classification(
        rc,
        sequences=sequences,
        progress_desc=desc,
        progress_position=0,
        precomputed_features=None,
    )


def _legacy_train_classification(
    rc: Any,
    sequences: jnp.ndarray,
    labels: jnp.ndarray,
    ridge_lambdas: Optional[Sequence[float]],
    num_classes: int,
    return_features: bool,
) -> Optional[jnp.ndarray]:
    raise NotImplementedError(
        "Analog quantum classification is legacy and not supported in dynamic_runner."
    )


def _legacy_predict_classification(
    rc: Any,
    sequences: jnp.ndarray,
    cached_features: Optional[jnp.ndarray],
    desc: str,
) -> jnp.ndarray:
    raise NotImplementedError(
        "Analog quantum classification is legacy and not supported in dynamic_runner."
    )


CLASSIFICATION_PIPELINES: Dict[str, Tuple[ClassTrainFn, ClassPredictFn]] = {
    "classical": (_train_classical_classification, _predict_classical_classification),
    "gatebased_quantum": (_train_quantum_classification, _predict_quantum_classification),
    "analog_quantum_legacy": (_legacy_train_classification, _legacy_predict_classification),
}


def _resolve_model_kind(model_name: str, model_type: str) -> str:
    """Resolve a coarse model kind for regression pipeline dispatch."""
    mt = (model_type or "").lower()
    name = (model_name or "").lower()
    if "analog" in name or "analog" in mt:
        return "analog_quantum_legacy"
    if "gatebased_quantum" in name:
        return "gatebased_quantum"
    if "quantum" in mt:
        # Fallback: treat non-analog quantum as gate-based
        return "gatebased_quantum"
    return "classical"


def _log_ridge_search(model: Any) -> Optional[list]:
    """Print ridge-search diagnostics if available and return the log."""
    ridge_log = getattr(model, "ridge_search_log", None)
    if ridge_log:
        print("Ridge λ grid search")
        for entry in ridge_log:
            lam = entry["lambda"]
            if "val_accuracy" in entry:
                print(f"  λ={lam:.2e} -> val Acc={entry['val_accuracy']:.4f}")
            elif "val_mse" in entry:
                print(f"  λ={lam:.2e} -> val MSE={entry['val_mse']:.6f}")
            elif "train_accuracy" in entry:
                print(f"  λ={lam:.2e} -> train Acc={entry['train_accuracy']:.4f}")
            elif "train_mse" in entry:
                print(f"  λ={lam:.2e} -> train MSE={entry['train_mse']:.6f}")
            else:
                print(f"  λ={lam:.2e}")
        best_lambda = getattr(model, "best_ridge_lambda", None)
        if best_lambda is not None:
            print(f"Selected λ={best_lambda:.2e}")
    return ridge_log


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
    print(f"Saved config snapshot -> {snapshot_path}")


def run_dynamic_experiment(
    dataset_name: str,
    model_name: str,
    training_name: str = "standard",
    show_training: bool = False,
    backend: Optional[str] = None,
    n_reservoir_override: Optional[int] = None,
) -> Tuple[Optional[float], float, Optional[float], float]:
    """Run an experiment with dynamic configuration.

    Args:
        dataset_name: Name of dataset config (e.g., 'sine_wave', 'lorenz')
        model_name: Name of model config (e.g., 'classic_standard', 'gatebased_quantum')
        training_name: Name of training config (default: 'standard')
        show_training: Whether to show training data in visualization
        backend: Compute backend ('cpu' or 'gpu')
        n_reservoir_override: Optional reservoir size override for classical models

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

    model_name_lower = model_name.lower()
    is_quantum_choice = "quantum" in model_name_lower
    if not is_quantum_choice and n_reservoir_override is None:
        raise ValueError(
            "n_reservoir_override is required for classical reservoir models. "
            "Pass --n-reservoir via CLI or script."
        )

    if n_reservoir_override is not None:
        try:
            override_value = int(n_reservoir_override)
        except (TypeError, ValueError) as exc:
            raise ValueError("n_reservoir_override must be an integer") from exc

        model_section = legacy_config.get("model", {})
        model_type = str(model_section.get("model_type", "reservoir")).lower()
        if "quantum" not in model_type:
            reservoir_section = legacy_config.get("reservoir")
            if isinstance(reservoir_section, dict):
                reservoir_section["n_reservoir"] = override_value
            params = model_section.setdefault("params", {})
            if isinstance(params, dict):
                params["n_reservoir"] = override_value

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
    dataset = prepare_experiment_data(experiment_config)

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
    model_factory = get_model_factory(model_type)

    data_config = demo_config.data_generation
    data_params = dict(data_config.params or {})
    data_n_input = data_config.n_input
    if data_n_input is None:
        data_n_input = data_params.get("n_input")
    data_n_output = data_config.n_output
    if data_n_output is None:
        data_n_output = data_params.get("n_output")

    raw_training = (demo_config.training.name == "raw_standard")
    state_agg_override = getattr(demo_config.training, "state_aggregation", None)

    if isinstance(dataset, ExperimentDatasetClassification):
        n_input = int(dataset.train_sequences.shape[-1])
        combined_labels = jnp.concatenate(
            [dataset.train_labels, dataset.test_labels]
        )
        if combined_labels.size > 0:
            num_classes = int(jnp.max(combined_labels).item()) + 1
        else:
            num_classes = 1

        if demo_config.reservoir is None:
            demo_config.reservoir = {}
        demo_config.reservoir['n_inputs'] = n_input
        demo_config.reservoir.setdefault('n_outputs', num_classes)
        default_agg = state_agg_override or ('last' if raw_training else 'mean')
        demo_config.reservoir.setdefault('state_aggregation', default_agg)
    else:
        if demo_config.reservoir is None:
            demo_config.reservoir = {}
        if data_n_input is not None:
            demo_config.reservoir.setdefault('n_inputs', int(data_n_input))
        if data_n_output is not None:
            demo_config.reservoir.setdefault('n_outputs', int(data_n_output))
        if state_agg_override is not None:
            demo_config.reservoir.setdefault('state_aggregation', state_agg_override)
        else:
            demo_config.reservoir.setdefault('state_aggregation', 'mean')

    if raw_training:
        if demo_config.reservoir is None:
            demo_config.reservoir = {}
        demo_config.reservoir.setdefault('use_preprocessing', False)
        demo_config.reservoir.setdefault('include_bias', False)
        demo_config.reservoir.setdefault('washout_steps', 0)

    dataset_name = demo_config.data_generation.name
    model_name = demo_config.model.name
    is_analog_model = "analog" in model_name or model_type == "analog_quantum"

    ridge_cfg = getattr(demo_config.training, "ridge_lambdas", None)
    ridge_defaults = list(ridge_cfg) if ridge_cfg else [1e-6, 1e-5, 1e-4, 1e-3]

    if is_analog_model:
        if demo_config.quantum_reservoir is None:
            raise ValueError("Analog quantum mode requires quantum_reservoir config")
        demo_config.quantum_reservoir.setdefault('ridge_lambdas', list(ridge_defaults))
        analog_base = _load_analog_quantum_config().get('params', {})
        config_dict: Dict[str, Any] = {"model_type": "analog_quantum"}
        config_dict.update(analog_base)
        config_dict.update(demo_config.quantum_reservoir)
        if data_n_input is not None:
            config_dict["n_inputs"] = int(data_n_input)
        if data_n_output is not None:
            config_dict["n_outputs"] = int(data_n_output)
        rc = model_factory.create_reservoir(
            'analog', config_dict, backend
        )
    elif quantum_mode or "quantum" in model_type:
        if demo_config.quantum_reservoir is None:
            raise ValueError("Quantum mode enabled but quantum_reservoir config is missing")
        demo_config.quantum_reservoir.setdefault('ridge_lambdas', list(ridge_defaults))
        quantum_base = _load_gatebased_quantum_config().get('params', {})
        basic_base = _load_shared_reservoir_config()
        config_sequence = [
            {'params': basic_base},
            {'params': quantum_base},
            demo_config.quantum_reservoir,
        ]
        if data_n_input is not None:
            demo_config.quantum_reservoir['n_inputs'] = int(data_n_input)
        if data_n_output is not None:
            demo_config.quantum_reservoir['n_outputs'] = int(data_n_output)
        rc = model_factory.create_reservoir(
            'quantum', config_sequence, backend
        )
    else:
        if demo_config.reservoir is None:
            raise ValueError("Classical mode requires reservoir config")
        demo_config.reservoir.setdefault('ridge_lambdas', list(ridge_defaults))
        basic_base = _load_shared_reservoir_config()
        config_sequence = [
            {'params': basic_base},
            demo_config.reservoir,
        ]
        rc = model_factory.create_reservoir(
            'classical', config_sequence, backend
        )

    reservoir_info = rc.get_reservoir_info()
    print(f"Reservoir情報: {reservoir_info}")

    is_quantum_model = quantum_mode or ("quantum" in (model_type or ""))
    n_reservoir: Optional[int] = None
    n_inputs_value: Optional[int] = None
    if not is_quantum_model:
        candidates: list[Any] = []
        if isinstance(reservoir_info, dict):
            candidates.append(reservoir_info.get("n_reservoir"))
        if hasattr(rc, "n_reservoir"):
            candidates.append(getattr(rc, "n_reservoir"))
        if demo_config.reservoir:
            candidates.append(demo_config.reservoir.get("n_reservoir"))

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                n_reservoir = int(candidate)
                break
            except (TypeError, ValueError):
                continue
        input_candidates: list[Any] = []
        if isinstance(reservoir_info, dict):
            input_candidates.append(reservoir_info.get("n_inputs"))
        if hasattr(rc, "n_inputs"):
            input_candidates.append(getattr(rc, "n_inputs"))
        if demo_config.reservoir:
            input_candidates.append(demo_config.reservoir.get("n_inputs"))

        for candidate in input_candidates:
            if candidate is None:
                continue
            try:
                n_inputs_value = int(candidate)
                break
            except (TypeError, ValueError):
                continue
    else:
        candidates: list[Any] = []
        if isinstance(reservoir_info, dict):
            candidates.append(reservoir_info.get("n_inputs"))
        if hasattr(rc, "n_inputs"):
            candidates.append(getattr(rc, "n_inputs"))
        if getattr(demo_config, "quantum_reservoir", None):
            input_cfg = demo_config.quantum_reservoir
            if isinstance(input_cfg, dict):
                candidates.append(input_cfg.get("n_inputs"))
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                n_inputs_value = int(candidate)
                break
            except (TypeError, ValueError):
                continue
    resolved_filename = demo_config.demo.filename
    filename_suffix_parts = []
    if is_analog_model:
        suffix = Path(resolved_filename).suffix or ".png"
        resolved_filename = f"{dataset_name}_aq_prediction{suffix}"
    elif quantum_mode or "quantum" in model_type:
        suffix = Path(resolved_filename).suffix or ".png"
        resolved_filename = f"{dataset_name}_gq_prediction{suffix}"

    if raw_training:
        filename_suffix_parts.append("raw")

    if n_reservoir is not None and not is_quantum_model:
        filename_suffix_parts.append(f"nr{n_reservoir}")

    plot_title = demo_config.demo.title

    if quantum_mode or "quantum" in model_type:
        qubit_candidates: list[Any] = []
        depth_candidates: list[Any] = []

        if isinstance(reservoir_info, dict):
            qubit_candidates.append(reservoir_info.get("n_qubits"))
            depth_candidates.append(reservoir_info.get("circuit_depth"))

        if hasattr(rc, "n_qubits"):
            qubit_candidates.append(getattr(rc, "n_qubits"))
        if hasattr(rc, "circuit_depth"):
            depth_candidates.append(getattr(rc, "circuit_depth"))

        if getattr(demo_config, "quantum_reservoir", None):
            quantum_cfg = demo_config.quantum_reservoir
            if isinstance(quantum_cfg, dict):
                qubit_candidates.append(quantum_cfg.get("n_qubits"))
                depth_candidates.append(quantum_cfg.get("circuit_depth"))

        n_qubits: Optional[int] = None
        for candidate in qubit_candidates:
            if candidate is None:
                continue
            try:
                n_qubits = int(candidate)
                break
            except (TypeError, ValueError):
                continue

        circuit_depth: Optional[int] = None
        for candidate in depth_candidates:
            if candidate is None:
                continue
            try:
                circuit_depth = int(candidate)
                break
            except (TypeError, ValueError):
                continue

        readout_features = reservoir_info.get("readout_feature_dim")
        readout_observables = reservoir_info.get("readout_observables")
        state_agg = str(reservoir_info.get("state_aggregation", "")).lower()

        if n_qubits is not None and readout_observables:
            from math import comb

            components: list[str] = []
            calculated_dim = 0
            for observable in readout_observables:
                obs = observable.upper()
                if obs in {"X", "Y", "Z"}:
                    calculated_dim += n_qubits
                    components.append(f"{n_qubits} {obs}")
                elif obs == "ZZ":
                    pairs = comb(int(n_qubits), 2)
                    calculated_dim += pairs
                    components.append(f"{pairs} ZZ")

            base_dim = readout_features or calculated_dim
            components_str = " + ".join(components) if components else f"{base_dim}"

            aggregated_dim = reservoir_info.get("feature_dim", base_dim)
            agg_note = ""
            if aggregated_dim != base_dim:
                agg_note = f"; after '{state_agg}' aggregation → {aggregated_dim}"

            print(
                f"🧮 Quantum feature dimension: {base_dim} ({components_str}){agg_note}"
            )
        elif n_qubits is not None and readout_features:
            print(f"🧮 Quantum feature dimension: {readout_features}")

        if n_qubits is not None or circuit_depth is not None:
            filename_path = Path(resolved_filename)
            suffix = filename_path.suffix or ""
            stem = filename_path.stem
            if n_qubits is not None and circuit_depth is not None:
                resolved_filename = f"{stem}_{n_qubits}_{circuit_depth}{suffix}"
            elif n_qubits is not None:
                resolved_filename = f"{stem}_{n_qubits}{suffix}"
            elif circuit_depth is not None:
                resolved_filename = f"{stem}_{circuit_depth}{suffix}"

    if n_inputs_value is not None:
        filename_suffix_parts.append(f"in{n_inputs_value}")

    if filename_suffix_parts:
        filename_path = Path(resolved_filename)
        suffix = filename_path.suffix or ""
        stem = filename_path.stem
        suffix_segment = "_".join(filename_suffix_parts)
        resolved_filename = f"{stem}_{suffix_segment}{suffix}"

    output_filename = resolved_filename

    ridge_cfg = demo_config.training.ridge_lambdas
    if ridge_cfg and isinstance(ridge_cfg, (list, tuple)) and len(ridge_cfg) == 3:
        # Format: [start, stop, num] or (start, stop, num)
        start, stop, num = ridge_cfg
        lambda_candidates = list(np.logspace(start, stop, int(num)))
    elif ridge_cfg:
        lambda_candidates = list(ridge_cfg)
    else:
        lambda_candidates = [1e-6, 1e-5, 1e-4, 1e-3]

    if isinstance(dataset, ExperimentDatasetClassification):
        train_count = int(dataset.train_sequences.shape[0])
        val_count = int(dataset.val_sequences.shape[0])
        test_count = int(dataset.test_sequences.shape[0])
        print(f"データ分割 → train: {train_count}, val: {val_count}, test: {test_count}")
        print("訓練中 (classification)...")

        model_kind = _resolve_model_kind(model_name, model_type or "")
        class_train_fn, class_predict_fn = CLASSIFICATION_PIPELINES.get(
            model_kind,
            CLASSIFICATION_PIPELINES["classical"],
        )

        train_features = class_train_fn(
            rc,
            dataset.train_sequences,
            dataset.train_labels,
            ridge_lambdas=lambda_candidates,
            num_classes=10,
            return_features=True,
        )
        ridge_log = _log_ridge_search(rc)

        print("予測中 (train/test/val inference)...")

        def _predict_with_cache(
            sequences,
            *,
            cached_features,
            desc: str,
        ):
            return class_predict_fn(
                rc,
                sequences,
                cached_features,
                desc,
            )

        train_logits = _predict_with_cache(
            dataset.train_sequences,
            cached_features=train_features,
            desc="Encoding train eval sequences",
        )
        test_logits = class_predict_fn(
            rc,
            dataset.test_sequences,
            cached_features=None,
            desc="Encoding test sequences",
        )
        val_logits = None
        if hasattr(dataset, "val_sequences") and dataset.val_sequences.size > 0:
            val_logits = class_predict_fn(
                rc,
                dataset.val_sequences,
                cached_features=None,
                desc="Encoding validation sequences",
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
        if val_mse is not None:
            metrics_caption["Val MSE"] = val_mse
            metrics_snapshot["val_mse"] = val_mse
        if val_accuracy is not None:
            metrics_caption["Val Acc"] = f"{val_accuracy:.4f}"
            metrics_snapshot["val_accuracy"] = val_accuracy

        best_lambda_attr = getattr(rc, "best_ridge_lambda", None)
        if best_lambda_attr is not None:
            best_lambda = float(best_lambda_attr)
            metrics_caption["Ridge λ"] = f"{best_lambda:.2e}"
            metrics_snapshot["best_ridge_lambda"] = best_lambda

        # 可視化 (classification)
        labels_arrays = [
            np.asarray(dataset.train_labels),
            np.asarray(dataset.test_labels),
            np.asarray(train_pred),
            np.asarray(test_pred),
        ]
        if hasattr(dataset, "val_labels") and dataset.val_labels.size > 0:
            labels_arrays.append(np.asarray(dataset.val_labels))
            if val_pred is not None:
                labels_arrays.append(np.asarray(val_pred))
        detected_classes = max(
            (int(arr.max()) if arr.size > 0 else -1) for arr in labels_arrays
        )
        class_count = max(detected_classes + 1, 10)
        class_names = [str(i) for i in range(class_count)]
        plot_classification_results(
            np.asarray(dataset.train_labels),
            np.asarray(dataset.test_labels),
            np.asarray(train_pred),
            np.asarray(test_pred),
            plot_title,
            output_filename,
            metrics_info=metrics_caption,
            class_names=class_names,
        )

        extra_results = {
            "classification": {
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "test_accuracy": test_accuracy,
                "reservoir_info": reservoir_info,
            }
        }
        _save_config_snapshot(
            demo_config,
            output_filename,
            metrics_snapshot,
            ridge_log,
            extra=extra_results,
        )

        return train_mse, test_mse, train_accuracy, test_accuracy

    print("訓練中...")
    model_kind = _resolve_model_kind(model_name, model_type or "")
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


def run_experiment_from_config(
    config_path: str,
    backend: Optional[str] = None,
    quantum_mode: bool = False,
) -> Tuple[Optional[float], float, Optional[float], float]:
    """High-level helper that loads config, prepares data, and runs the experiment."""

    demo_config = ExperimentConfig.from_json(config_path)
    dataset = prepare_experiment_data(demo_config)
    return run_experiment(demo_config, dataset, backend=backend, quantum_mode=quantum_mode)
