"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/factory.py
STEP 4, 5, 6 read README.md
Global entry point for model creation. Delegates to specialized factories.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from reservoir.models.nn.factory import NNModelFactory
from reservoir.models.nn.fnn import FNNModel
from reservoir.training.presets import TrainingConfig
from reservoir.core.identifiers import Model
from reservoir.models.config import ClassicalReservoirConfig, DistillationConfig, FNNConfig
from reservoir.models.presets import PipelineConfig
from reservoir.models.distillation.factory import DistillationFactory
from reservoir.models.reservoir.factory import ReservoirFactory


class ModelFactory:
    """Router that delegates model construction to specialized factories."""

    @staticmethod
    def create_model(
        config: PipelineConfig,
        training: TrainingConfig = None,
        input_dim: int = None,
        output_dim: int = None,
        input_shape: tuple[int, ...] = None,
    ) -> Any:

        if input_dim is None or input_dim <= 0:
            raise ValueError("ModelFactory.create_model requires a positive input_dim.")
        if output_dim is None or output_dim <= 0:
            raise ValueError("ModelFactory.create_model requires a positive output_dim.")

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

        if pipeline_enum == Model.FNN:
            if not isinstance(config.model, FNNConfig):
                raise TypeError(f"FNN pipeline requires FNNConfig, got {type(config.model)}.")
            model = FNNModel(
                model_config=config.model,
                training_config=training_cfg,
                input_dim=input_dim,
                output_dim=output_dim,
            )
            # Attach topology metadata for FNN
            flattened_dim = int(input_dim)
            topo_meta: Dict[str, Any] = {
                "type": pipeline_enum.value.upper(),
                "shapes": {
                    "input": input_shape,
                    "projected": input_shape,
                    "adapter": (flattened_dim,),
                    "internal": tuple(config.model.hidden_layers) if config.model.hidden_layers else None,
                    "feature": (output_dim,),
                    "output": (output_dim,),
                },
                "details": {
                    "student_layers": tuple(config.model.hidden_layers) if config.model.hidden_layers else None,
                    "structure": "Flatten -> FNN -> Output",
                    "agg_mode": "None",
                    "readout": "None",
                },
            }
            model.topology_meta = topo_meta
            return model


        raise ValueError(f"Unsupported model_type: {pipeline_enum}")
