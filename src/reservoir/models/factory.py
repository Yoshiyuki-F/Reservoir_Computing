"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/factory.py
Global entry point for model creation. Delegates to specialized factories.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

from reservoir.training.presets import TrainingConfig, get_training_preset
from reservoir.core.identifiers import Dataset, Pipeline, RunConfig, TaskType, Preprocessing, AggregationMode
from reservoir.data.presets import DATASET_REGISTRY, DatasetPreset
from reservoir.models.presets import get_model_preset, DistillationConfig
from reservoir.models.nn.factory import NNModelFactory
from reservoir.models.distillation.factory import DistillationFactory
from reservoir.models.reservoir.factory import ReservoirFactory


class ModelFactory:
    """Router that delegates model construction to specialized factories."""

    @staticmethod
    def create_model(
        config: RunConfig,
        *,
        dataset_preset: Optional[DatasetPreset] = None,
        training: Optional[TrainingConfig] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        input_shape: Optional[tuple[int, ...]] = None,
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

        if pipeline_enum in {Pipeline.CLASSICAL_RESERVOIR, Pipeline.QUANTUM_GATE_BASED, Pipeline.QUANTUM_ANALOG}:
            reservoir_cfg = ModelFactory._resolve_reservoir_params(pipeline_enum)
            return ReservoirFactory.create_pipeline(
                reservoir_cfg,
                input_dim=resolved_input_dim,
                output_dim=resolved_output_dim,
                use_preprocessing=use_preprocessing,
                input_shape=input_shape,
                pipeline=pipeline_enum,
            )

        if pipeline_enum == Pipeline.FNN_DISTILLATION:
            distill_cfg = ModelFactory._resolve_distillation_config(pipeline_enum)
            return DistillationFactory.create_model(
                distill_cfg,
                training=training_cfg,
                input_dim=resolved_input_dim,
                output_dim=resolved_output_dim,
                input_shape=input_shape,
                use_preprocessing=use_preprocessing,
            )

        if pipeline_enum == Pipeline.FNN:
            layer_dims = [resolved_input_dim, resolved_output_dim]
            fnn_cfg = {"layer_dims": layer_dims}
            model = NNModelFactory.create_fnn(fnn_cfg, training_cfg)
            in_shape = input_shape or (1, resolved_input_dim)
            projected_shape = (resolved_input_dim,)
            internal_shape = (layer_dims[-2],) if len(layer_dims) >= 2 else None
            feature_units = layer_dims[-2] if len(layer_dims) >= 2 else resolved_input_dim
            topo_meta = {
                "type": pipeline_enum.value.upper(),
                "shapes": {
                    "input": in_shape,
                    "preprocessed": in_shape,
                    "projected": projected_shape,
                    "internal": internal_shape,
                    "feature": (feature_units,),
                    "output": (resolved_output_dim,),
                },
                "details": {"preprocess": None, "agg_mode": None, "student_layers": layer_dims[1:-1]},
            }
            setattr(model, "topology_meta", topo_meta)
            return model

        if pipeline_enum == Pipeline.RNN_DISTILLATION:
            model_cfg = {
                "input_dim": resolved_input_dim,
                "hidden_dim": resolved_output_dim,
                "output_dim": resolved_output_dim,
            }
            return NNModelFactory.create_rnn(model_cfg, training_cfg)

        raise ValueError(f"Unsupported model_type: {pipeline_enum}")

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
    def _resolve_distillation_config(pipeline: Pipeline) -> DistillationConfig:
        preset = get_model_preset(pipeline)
        if preset is None:
            raise ValueError(f"Model preset '{pipeline}' not found for distillation.")
        distill_cfg = preset.config
        if not isinstance(distill_cfg, DistillationConfig):
            raise ValueError(f"Preset '{preset.name}' must use DistillationConfig for distillation workflows.")
        distill_cfg.teacher.validate(context=f"preset.{preset.name}.teacher")
        return distill_cfg

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
