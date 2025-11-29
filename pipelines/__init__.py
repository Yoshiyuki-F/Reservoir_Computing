"""Data pipelines package.

Top-level modules focus on experiment workflows (classical/quantum/FNN).
Generic helpers live under `reservoir.utils` and `reservoir.data`.
Imports here are lazy to avoid circular dependencies with models that
depend on individual pipeline modules.
"""

__all__ = [
    "UniversalPipeline",
    "run_pipeline",
    "run_rnn_pipeline",
    "run_fnn_pipeline",
    "run_reservoir_pipeline",
    "load_dataset",
    "prepare_experiment_data",
    "ExperimentDataset",
]


def __getattr__(name):
    if name == "UniversalPipeline":
        from .generic_runner import UniversalPipeline

        return UniversalPipeline
    if name in {"prepare_experiment_data", "ExperimentDataset"}:
        from reservoir.data.data_preparation import prepare_experiment_data, ExperimentDataset

        return {"prepare_experiment_data": prepare_experiment_data, "ExperimentDataset": ExperimentDataset}[
            name
        ]
    if name in {"run_pipeline", "run_rnn_pipeline", "run_fnn_pipeline", "run_reservoir_pipeline", "load_dataset"}:
        from .run import run_pipeline, run_rnn_pipeline, run_fnn_pipeline, run_reservoir_pipeline, load_dataset

        return {
            "run_pipeline": run_pipeline,
            "run_rnn_pipeline": run_rnn_pipeline,
            "run_fnn_pipeline": run_fnn_pipeline,
            "run_reservoir_pipeline": run_reservoir_pipeline,
            "load_dataset": load_dataset,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
