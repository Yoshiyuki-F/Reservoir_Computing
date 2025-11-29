"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/base.py
Base abstractions for models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp


class BaseModel(ABC):
    """Abstract base class for all models."""

    @abstractmethod
    def train(self, X: jnp.ndarray, y: jnp.ndarray) -> Optional[Dict[str, Any]]:
        """Train the model.

        Args:
            X: Input data
            y: Target data

        Returns:
            Training metrics/state if available. Can be None for models that only
            expose inference metrics.
        """
        pass

    @abstractmethod
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Make predictions.

        Args:
            X: Input data

        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            X: Input data
            y: Target data

        Returns:
            Evaluation metrics
        """
        pass


class ModelFactory(ABC):
    """Abstract factory for creating models."""

    @staticmethod
    @abstractmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        """Create a model instance from configuration.

        Args:
            config: Model configuration

        Returns:
            Model instance
        """
        pass


class BaseRunner(ABC):
    """Abstract base class for experiment runners."""

    @abstractmethod
    def run_experiment(self, config_path: str, **kwargs) -> Tuple[float, float, float, float]:
        """Run an experiment.

        Args:
            config_path: Path to configuration file
            **kwargs: Additional arguments

        Returns:
            Tuple of (train_mse, test_mse, train_mae, test_mae)
        """
        pass
