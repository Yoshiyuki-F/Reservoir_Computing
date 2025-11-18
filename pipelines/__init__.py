"""Data pipelines package.

Top-level modules focus on experiment workflows (classical/quantum/FNN).
Generic helpers live under `core_lib.utils`.
"""

from .data_preparation import prepare_experiment_data, ExperimentDataset  # noqa: F401

__all__ = [
    "prepare_experiment_data",
    "ExperimentDataset",
]
