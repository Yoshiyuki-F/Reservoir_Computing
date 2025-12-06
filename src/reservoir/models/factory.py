"""src/reservoir/models/factory.py
Global entry point for model creation. Routes requests to specialized factories.
"""
from typing import Any, Dict

from reservoir.training.presets import TrainingConfig
from reservoir.core.identifiers import Pipeline
from reservoir.models.nn.factory import NNModelFactory
from reservoir.models.distillation.factory import DistillationFactory
from reservoir.models.reservoir.factory import ReservoirFactory


class ModelFactory:
    """Router that decides which model to build based on configuration context."""

    @staticmethod
    def create_model(config: Dict[str, Any]) -> Any:
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
