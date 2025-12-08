"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/factory.py
STEP 4, 5, 6 read README.md
Global entry point for model creation. Delegates to specialized factories.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

from reservoir.training.presets import TrainingConfig, get_training_preset
from reservoir.core.identifiers import Dataset, Model, TaskType
from reservoir.data.presets import DATASET_REGISTRY, DatasetPreset
from reservoir.models.config import ClassicalReservoirConfig, DistillationConfig
from reservoir.models.presets import PipelineConfig
from reservoir.models.distillation.factory import DistillationFactory
from reservoir.models.reservoir.factory import ReservoirFactory


class ModelFactory:
    """Router that delegates model construction to specialized factories."""

    @staticmethod
    def create_model(
        config: PipelineConfig,
        dataset_preset: DatasetPreset = None,
        training: TrainingConfig = None,
        input_dim: int = None,
        output_dim: int = None,
        input_shape: tuple[int, ...] = None,
    ) -> Any:

        preset = dataset_preset
        if input_dim <= 0:
            raise ValueError(f"Dataset '{preset.name}' must define n_input > 0.")
        if output_dim <= 0:
            raise ValueError(f"Dataset '{preset.name}' must define n_output > 0.")

        training_cfg = training
        pipeline_enum = config.model_type

        if pipeline_enum in {Model.CLASSICAL_RESERVOIR, Model.QUANTUM_GATE_BASED, Model.QUANTUM_ANALOG}:
            if not isinstance(config.model, ClassicalReservoirConfig):
                raise TypeError(f"Reservoir pipelines require ClassicalReservoirConfig, got {type(config.model)}.")

            return ReservoirFactory.create_pipeline(
                pipeline_config=config,
                projected_input_dim=input_dim,
                output_dim=output_dim,
                input_shape=input_shape,
            )

        if pipeline_enum == Model.FNN_DISTILLATION:
            if not isinstance(config.model, DistillationConfig):
                raise TypeError(f"FNN_DISTILLATION pipeline requires DistillationConfig, got {type(config.model)}.")
            return DistillationFactory.create_model(
                distillation_config=config.model,
                training=training_cfg,
                input_dim=input_dim,
                output_dim=output_dim,
                input_shape=input_shape,
            )

        raise ValueError(f"Unsupported model_type: {pipeline_enum}")

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_training(task_type: TaskType) -> TrainingConfig:
        preset = get_training_preset("standard")
        return replace(preset, classification=task_type is TaskType.CLASSIFICATION)

    @staticmethod
    def _get_dataset_preset(dataset: Optional[Dataset]) -> DatasetPreset:
        if dataset is None:
            raise ValueError("Dataset must be provided to resolve presets.")
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Dataset preset lookup requires Dataset Enum, got {type(dataset)}.")
        preset = DATASET_REGISTRY.get(dataset)
        if preset is None:
            raise ValueError(f"Dataset preset '{dataset}' not found in registry.")
        if preset.config.n_input is None or preset.config.n_output is None:
            raise ValueError(f"Dataset preset '{dataset}' must define n_input and n_output.")
        return preset
