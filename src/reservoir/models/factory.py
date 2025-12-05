"""src/reservoir/models/factory.py
Global entry point for model creation. Routes requests to specialized factories.
"""
from typing import Any, Dict

from reservoir.training.presets import TrainingConfig
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

        # Reservoir family
        if model_type in ("reservoir", "classical", "quantum_gate_based", "quantum_analog"):
            return ReservoirFactory.create_model(config)

        # RNN (no distillation path here)
        if model_type == "rnn":
            return NNModelFactory.create_rnn(model_cfg, training_cfg)

        # FNN (pure or distillation)
        if model_type == "fnn":
            has_reservoir = bool(config.get("reservoir") or config.get("reservoir_params"))
            if has_reservoir:
                return DistillationFactory.create(model_cfg, training_cfg, config)
            return NNModelFactory.create_fnn(model_cfg, training_cfg)

        raise ValueError(f"Unsupported model_type: {model_type}")
