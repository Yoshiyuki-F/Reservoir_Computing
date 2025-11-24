"""Builders for reservoir models and related configuration handling."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import jax.numpy as jnp

from core_lib.core import ExperimentConfig
from core_lib.data import ExperimentDataset, ExperimentDatasetClassification
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
def _load_gatebased_quantum_config() -> Dict[str, Any]:
    return _load_config_json('presets/models/gatebased_quantum.json')


@dataclass
class ReservoirBuildResult:
    rc: Any
    reservoir_info: Dict[str, Any]
    model_type: str
    is_quantum_model: bool
    n_hiddenLayer: Optional[int]
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
    n_hiddenLayer: Optional[int] = None
    n_inputs_value: Optional[int] = None
    if not is_quantum_model:
        candidates: list[Any] = []
        if isinstance(reservoir_info, dict):
            candidates.append(reservoir_info.get("n_hiddenLayer"))
        if hasattr(rc, "n_hiddenLayer"):
            candidates.append(getattr(rc, "n_hiddenLayer"))
        if demo_config.reservoir:
            candidates.append(demo_config.reservoir.get("n_hiddenLayer"))

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                n_hiddenLayer = int(candidate)
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
        n_hiddenLayer=n_hiddenLayer,
        n_inputs_value=n_inputs_value,
        raw_training=raw_training,
    )
