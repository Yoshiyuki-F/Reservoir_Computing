# /home/yoshi/PycharmProjects/Reservoir/src/reservoir/pipelines/config_builder.py
from __future__ import annotations
from reservoir.core.identifiers import Dataset, Pipeline, Preprocessing, ReadOutType, RunConfig


def build_run_config(*, preset_name: str, dataset_name: str) -> RunConfig:
    """
    Strict V2 Config Builder.
    Accepts ONLY the model preset and dataset identifiers, validating eagerly.
    """
    if not dataset_name:
        raise ValueError("dataset_name is required.")
    if not preset_name:
        raise ValueError("preset_name is required.")

    pipeline_enum = Pipeline(preset_name)
    dataset_enum = Dataset(dataset_name)
    task_type = dataset_enum.task_type

    return RunConfig(
        dataset=dataset_enum,
        model_type=pipeline_enum,
        task_type=task_type,
    )


__all__ = ["build_run_config"]
