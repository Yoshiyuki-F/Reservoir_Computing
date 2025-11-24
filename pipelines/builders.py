"""Builders for reservoir models and related configuration handling."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import jax.numpy as jnp

from core_lib.core import ExperimentConfig
from core_lib.data import ExperimentDataset, ExperimentDatasetClassification
from core_lib.models.fnn import FNNPipelineConfig
from pipelines.dispatchers import get_model_factory


@lru_cache()
def _load_config_json(relative_path: str) -> Dict[str, Any]:
    path = Path(relative_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / relative_path
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


@lru_cache()
def _load_shared_reservoir_config() -> Dict[str, Any]:
    return _load_config_json('presets/models/shared_reservoir_params.json')


@lru_cache()
def _load_gate_based_quantum_config() -> Dict[str, Any]:
    return _load_config_json('presets/models/gate_based_quantum.json')


@lru_cache()
def _load_training_preset(training_name: str) -> Dict[str, Any]:
    """Load a training preset JSON from presets/training/, falling back to standard."""
    base_dir = Path(__file__).resolve().parents[1] / "presets" / "training"
    preset_path = base_dir / f"{training_name}.json"
    if preset_path.exists():
        return json.loads(preset_path.read_text())

    fallback_path = base_dir / "standard.json"
    if training_name != "standard" and fallback_path.exists():
        print(f"Warning: training preset '{training_name}' not found. Falling back to 'standard'.")
        return json.loads(fallback_path.read_text())

    print(f"Warning: training preset '{training_name}' not found. Using default values.")
    return {}


@dataclass
class ReservoirBuildResult:
    rc: Any
    reservoir_info: Dict[str, Any]
    model_type: str
    is_quantum_model: bool
    n_hidden_layer: Optional[int]
    n_inputs_value: Optional[int]
    raw_training: bool


def build_reservoir_model(
    demo_config: ExperimentConfig,
    dataset: ExperimentDataset,
    *,
    backend: Optional[str] = None,
    quantum_mode: bool = False,
) -> ReservoirBuildResult:
    """Construct reservoir model and return build metadata."""

    resolved_model_type = "quantum" if quantum_mode else "classical"
    model_factory = get_model_factory(resolved_model_type)

    data_config = demo_config.data_generation
    data_params = dict(data_config.params or {})
    data_n_input = data_config.n_input or data_params.get("n_input")
    data_n_output = data_config.n_output or data_params.get("n_output")

    raw_training = (demo_config.training.name == "raw_standard")
    state_agg_override = getattr(demo_config.training, "state_aggregation", None)

    if isinstance(dataset, ExperimentDatasetClassification):
        n_input = int(dataset.train_sequences.shape[-1])
        combined_labels = jnp.concatenate([dataset.train_labels, dataset.test_labels])
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

    ridge_cfg = getattr(demo_config.training, "ridge_lambdas", None)
    ridge_defaults = list(ridge_cfg) if ridge_cfg else [1e-6, 1e-5, 1e-4, 1e-3]

    if quantum_mode or "quantum" in resolved_model_type:
        if demo_config.quantum_reservoir is None:
            raise ValueError("Quantum mode enabled but quantum_reservoir config is missing")
        demo_config.quantum_reservoir.setdefault('ridge_lambdas', list(ridge_defaults))
        quantum_base = _load_gate_based_quantum_config().get('params', {})
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
        rc = model_factory.create_reservoir('quantum', config_sequence, backend)
    else:
        if demo_config.reservoir is None:
            raise ValueError("Classical mode requires reservoir config")
        demo_config.reservoir.setdefault('ridge_lambdas', list(ridge_defaults))
        basic_base = _load_shared_reservoir_config()
        config_sequence = [
            {'params': basic_base},
            demo_config.reservoir,
        ]
        rc = model_factory.create_reservoir('classical', config_sequence, backend)

    reservoir_info = rc.get_reservoir_info()
    print(f"Reservoir情報: {reservoir_info}")

    is_quantum_model = quantum_mode or ("quantum" in resolved_model_type)
    n_hidden_layer: Optional[int] = None
    n_inputs_value: Optional[int] = None
    if not is_quantum_model:
        candidates: list[Any] = []
        if isinstance(reservoir_info, dict):
            candidates.append(reservoir_info.get("n_hidden_layer"))
        if hasattr(rc, "n_hidden_layer"):
            candidates.append(getattr(rc, "n_hidden_layer"))
        if demo_config.reservoir:
            candidates.append(demo_config.reservoir.get("n_hidden_layer"))

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                n_hidden_layer = int(candidate)
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
        candidates = []
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

    return ReservoirBuildResult(
        rc=rc,
        reservoir_info=reservoir_info,
        model_type=resolved_model_type,
        is_quantum_model=is_quantum_model,
        n_hidden_layer=n_hidden_layer,
        n_inputs_value=n_inputs_value,
        raw_training=raw_training,
    )


def build_fnn_config(
    dataset_name: str,
    n_hidden: int,
    reservoir_size: Optional[int] = None,
    training_name: str = "standard",
) -> FNNPipelineConfig:
    """Construct an FNNPipelineConfig using presets/training values."""

    train_preset = _load_training_preset(training_name)
    learning_rate = float(train_preset.get("learning_rate", 0.001))
    batch_size = int(train_preset.get("batch_size", 128))
    num_epochs = int(train_preset.get("num_epochs", 20))
    ridge_lambdas = train_preset.get("ridge_lambdas", [-7, 7, 15])

    input_dim = 784
    output_dim = 10
    suffix = f"h{int(n_hidden)}"

    if reservoir_size is not None:
        res_size = int(reservoir_size)
        suffix = f"{suffix}_vs_res{res_size}"
        input_dim = 28 * res_size
        output_dim = res_size

    base_dir = Path("outputs") / dataset_name
    base_dir.mkdir(parents=True, exist_ok=True)
    weights_path = base_dir / f"mnist_fnn_raw_{suffix}.msgpack"

    cfg_dict = {
        "model": {
            "layer_dims": [input_dim, int(n_hidden), output_dim]
        },
        "training": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "weights_path": str(weights_path),
        },
        "ridge_lambdas": ridge_lambdas,
        "use_preprocessing": False,
    }

    return FNNPipelineConfig(**cfg_dict)
