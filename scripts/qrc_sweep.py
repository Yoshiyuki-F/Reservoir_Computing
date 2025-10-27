#!/usr/bin/env python3
"""Grid sweep helper for quantum reservoir configurations.

Iterates over combinations of qubit counts and circuit depths, running the
standard Mackey-Glass quantum experiment for each pair. Results (plots,
snapshots, metrics) are emitted via the existing experiment pipeline.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Iterable

from configs.core import ConfigComposer, ComposedConfig, ExperimentConfig
from experiments.dynamic_runner import run_experiment
from pipelines.data_preparation import prepare_experiment_data


def generate_variants(
    base_composed: ComposedConfig,
    qubits: Iterable[int],
    depths: Iterable[int],
):
    for n_qubits, circuit_depth in product(qubits, depths):
        model_cfg = deepcopy(base_composed.model)
        model_cfg["n_qubits"] = n_qubits
        model_cfg["circuit_depth"] = circuit_depth
        model_cfg.setdefault("name", "quantum_standard")

        viz_cfg = deepcopy(base_composed.visualization)
        viz_cfg["title"] = f"Quantum MG q={n_qubits} d={circuit_depth}"
        viz_cfg["filename"] = f"mackey_glass_q{n_qubits}_d{circuit_depth}_prediction.png"

        yield ComposedConfig(
            dataset=deepcopy(base_composed.dataset),
            model=model_cfg,
            training=deepcopy(base_composed.training),
            visualization=viz_cfg,
            experiment_name=f"mg_quantum_q{n_qubits}_d{circuit_depth}",
            experiment_description=base_composed.experiment_description,
        )


def main() -> None:
    composer = ConfigComposer()

    base_experiment = {
        "name": "mg_quantum_sweep",
        "description": "Quantum Mackey-Glass sweep",
        "dataset": "mackey_glass",
        "model": "quantum_standard",
        "training": "standard",
        "visualization": {
            "show_training": False,
            "add_test_zoom": True,
        },
    }

    base_composed = composer.compose_experiment(base_experiment)

    qubits = [4, 6, 8]
    depths = [2, 4, 6]

    for variant in generate_variants(base_composed, qubits, depths):
        legacy_config = composer.compose_legacy_format(variant)
        experiment_config = ExperimentConfig(**legacy_config)

        dataset = prepare_experiment_data(experiment_config, quantum_mode=True)
        run_experiment(
            experiment_config,
            dataset,
            backend='gpu',
            quantum_mode=True,
            model_type=experiment_config.model.name,
        )


if __name__ == "__main__":
    main()
