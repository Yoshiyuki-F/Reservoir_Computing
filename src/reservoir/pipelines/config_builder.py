# /home/yoshi/PycharmProjects/Reservoir/src/reservoir/pipelines/config_builder.py
from __future__ import annotations

from typing import Tuple

from reservoir.core.identifiers import Model, Dataset
from reservoir.models.presets import PipelineConfig
from reservoir.models.presets import MODEL_DEFINITIONS


def build_run_config(*, preset_name: str, dataset_name: str) -> Tuple[PipelineConfig, Dataset]:
    """
    Strict V2 Config Builder.
    Accepts ONLY the model preset and dataset identifiers, validating eagerly.
    """
    if not dataset_name:
        raise ValueError("dataset_name is required.")
    if not preset_name:
        raise ValueError("preset_name is required.")

    model = Model(preset_name)
    dataset = Dataset(dataset_name)

    return MODEL_DEFINITIONS[model], dataset


__all__ = ["build_run_config"]
