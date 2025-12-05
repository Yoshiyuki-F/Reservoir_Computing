# /home/yoshi/PycharmProjects/Reservoir/src/reservoir/core/config_builder.py
from __future__ import annotations

from typing import Dict


def build_run_config(*, preset_name: str, dataset_name: str) -> Dict[str, str]:
    """
    Strict V2 Config Builder.
    Accepts ONLY the model preset and dataset identifiers.
    """
    if not dataset_name:
        raise ValueError("dataset_name is required.")
    if not preset_name:
        raise ValueError("preset_name is required.")

    return {
        "model_type": preset_name.lower(),
        "dataset": dataset_name.lower(),
        "training_preset": "standard",
    }


__all__ = ["build_run_config"]
