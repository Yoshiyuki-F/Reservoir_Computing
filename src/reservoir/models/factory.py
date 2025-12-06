"""src/reservoir/models/factory.py
Global entry point for model creation. Routes requests to specialized factories.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Union

from reservoir.training.presets import TrainingConfig, get_training_preset
from reservoir.core.identifiers import Dataset, Pipeline, RunConfig, TaskType, Preprocessing
from reservoir.data.presets import DATASET_REGISTRY, DatasetPreset
from reservoir.models.presets import (
    get_model_preset,
    DistillationConfig,
    MODEL_PRESETS,
)
from reservoir.models.nn.factory import NNModelFactory
from reservoir.models.distillation.factory import DistillationFactory
from reservoir.models.reservoir.factory import ReservoirFactory


class ModelFactory:
    """Router that decides which model to build based on configuration context."""

    @staticmethod
    def create_model(
        config: Union[RunConfig, Dict[str, Any]],
        *,
        dataset_preset: Optional[DatasetPreset] = None,
        training: Optional[TrainingConfig] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ) -> Any:
        """
        Primary entrypoint. Accepts modern RunConfig (preferred) or legacy dictionaries.
        """
        if isinstance(config, RunConfig):
            return ModelFactory._create_from_run_config(
                config,
                dataset_preset=dataset_preset,
                training=training,
                input_dim=input_dim,
                output_dim=output_dim,
            )
        return ModelFactory._create_from_dict(config)

    # ------------------------------------------------------------------ #
    # RunConfig path                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _create_from_run_config(
        config: RunConfig,
        *,
        dataset_preset: Optional[DatasetPreset],
        training: Optional[TrainingConfig],
        input_dim: Optional[int],
        output_dim: Optional[int],
    ) -> Any:
        preset = dataset_preset or ModelFactory._get_dataset_preset(config.dataset)
        data_cfg = preset.config
        resolved_input_dim = int(input_dim or data_cfg.n_input or 0)
        resolved_output_dim = int(output_dim or data_cfg.n_output or 0)
        if resolved_input_dim <= 0:
            raise ValueError(f"Dataset '{preset.name}' must define n_input > 0.")
        if resolved_output_dim <= 0:
            raise ValueError(f"Dataset '{preset.name}' must define n_output > 0.")

        training_cfg = training or ModelFactory._build_training(config.task_type)
        use_preprocessing = config.preprocessing != Preprocessing.RAW

        pipeline_enum = config.model_type
        if pipeline_enum in {
            Pipeline.CLASSICAL_RESERVOIR,
            Pipeline.QUANTUM_GATE_BASED,
            Pipeline.QUANTUM_ANALOG,
        }:
            reservoir_cfg = ModelFactory._resolve_reservoir_params(pipeline_enum)
            factory_cfg = {
                "type": pipeline_enum,
                "reservoir": reservoir_cfg,
                "input_dim": resolved_input_dim,
                "use_preprocessing": use_preprocessing,
                "training": training_cfg,
            }
            return ReservoirFactory.create_model(factory_cfg)

        if pipeline_enum == Pipeline.FNN_DISTILLATION:
            distill_ctx = ModelFactory._resolve_distillation_from_preset(pipeline_enum)
            if distill_ctx is None:
                raise ValueError("FNN_DISTILLATION requires a DistillationConfig preset.")
            teacher_params = distill_ctx["teacher_params"]
            student_hidden = distill_ctx["student_hidden"]
            fnn_cfg = {
                "layer_dims": [resolved_input_dim, *student_hidden, int(teacher_params["n_units"])],
            }
            distill_config = {
                "reservoir": teacher_params,
                "input_dim": resolved_input_dim,
                "student_hidden": student_hidden,
            }
            return DistillationFactory.create(fnn_cfg, training_cfg, distill_config)

        if pipeline_enum == Pipeline.FNN:
            fnn_cfg = {"layer_dims": [resolved_input_dim, resolved_output_dim]}
            return NNModelFactory.create_fnn(fnn_cfg, training_cfg)

        if pipeline_enum == Pipeline.RNN_DISTILLATION:
            model_cfg = {
                "input_dim": resolved_input_dim,
                "hidden_dim": resolved_output_dim,
                "output_dim": resolved_output_dim,
            }
            return NNModelFactory.create_rnn(model_cfg, training_cfg)

        raise ValueError(f"Unsupported model_type: {pipeline_enum}")

    # ------------------------------------------------------------------ #
    # Legacy dict path                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _create_from_dict(config: Dict[str, Any]) -> Any:
        model_type = config.get("type")
        model_cfg = dict(config.get("model", {}) or {})
        training_cfg = config.get("training")

        if not isinstance(training_cfg, TrainingConfig):
            raise TypeError("ModelFactory expects 'training' to be a TrainingConfig instance.")

        pipeline_enum = model_type if isinstance(model_type, Pipeline) else Pipeline(str(model_type))

        # Reservoir family
        if pipeline_enum in {
            Pipeline.CLASSICAL_RESERVOIR,
            Pipeline.QUANTUM_GATE_BASED,
            Pipeline.QUANTUM_ANALOG,
        }:
            creator = getattr(ReservoirFactory, "create", ReservoirFactory.create_model)
            try:
                return creator(config)
            except TypeError:
                input_shape = config.get("input_dim")
                reservoir_type = config.get("reservoir_type") or pipeline_enum.value
                backend = config.get("backend")
                return creator(config, input_shape, reservoir_type=reservoir_type, backend=backend)

        # RNN (no distillation path here)
        if pipeline_enum == Pipeline.RNN_DISTILLATION:
            return NNModelFactory.create_rnn(model_cfg, training_cfg)

        # FNN (pure or distillation)
        if pipeline_enum in {Pipeline.FNN, Pipeline.FNN_DISTILLATION}:
            has_reservoir = bool(config.get("reservoir") or config.get("reservoir_params"))
            if has_reservoir:
                return DistillationFactory.create(model_cfg, training_cfg, config)
            return NNModelFactory.create_fnn(model_cfg, training_cfg)

        raise ValueError(f"Unsupported model_type: {model_type}")

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_training(task_type: TaskType) -> TrainingConfig:
        preset = get_training_preset("standard")
        return replace(preset, classification=task_type is TaskType.CLASSIFICATION)

    @staticmethod
    def _resolve_reservoir_params(pipeline: Pipeline) -> Dict[str, Any]:
        preset = get_model_preset(pipeline)
        if preset.config is None:
            raise ValueError(f"Model preset '{pipeline.value}' is missing reservoir configuration.")
        if isinstance(preset.config, DistillationConfig):
            teacher_cfg = preset.config.teacher
            teacher_cfg.validate(context=f"{pipeline.value}.teacher")
            return teacher_cfg.to_dict()
        preset.config.validate(context=pipeline.value)
        return preset.config.to_dict()

    @staticmethod
    def _resolve_distillation_from_preset(pipeline: Pipeline) -> Optional[Dict[str, Any]]:
        preset = MODEL_PRESETS.get(pipeline)
        if preset is None:
            return None
        distill_cfg = preset.config if isinstance(preset.config, DistillationConfig) else (
            preset.params.get("distillation") if preset.params else None
        )
        if distill_cfg is None:
            return None
        if not isinstance(distill_cfg, DistillationConfig):
            raise ValueError(f"Preset '{preset.name}' must use DistillationConfig for distillation workflows.")

        teacher_cfg = distill_cfg.teacher
        teacher_cfg.validate(context=f"preset.{preset.name}.teacher")
        teacher_params = teacher_cfg.to_dict()

        student_hidden = tuple(int(v) for v in distill_cfg.student_hidden_layers)
        if not student_hidden:
            raise ValueError(f"Preset '{preset.name}' distillation must define at least one student hidden layer.")

        return {
            "preset": preset,
            "teacher_params": teacher_params,
            "student_hidden": student_hidden,
        }

    @staticmethod
    def _get_dataset_preset(dataset: Dataset) -> DatasetPreset:
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Dataset preset lookup requires Dataset Enum, got {type(dataset)}.")
        preset = DATASET_REGISTRY.get(dataset)
        if preset is None:
            raise ValueError(f"Dataset preset '{dataset}' not found in registry.")
        if preset.config.n_input is None or preset.config.n_output is None:
            raise ValueError(f"Dataset preset '{dataset}' must define n_input and n_output.")
        return preset
