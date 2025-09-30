"""
Abstract base class for Reservoir Computing implementations.

This module defines the common interface for both classical and quantum
reservoir computers, enabling unified usage and experimentation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union

from pipelines.jax_config import ensure_x64_enabled

ensure_x64_enabled()

import jax.numpy as jnp


class BaseReservoirComputer(ABC):
    """Abstract base class for all reservoir computer implementations.

    This class defines the essential interface that both classical and quantum
    reservoir computers must implement, allowing for seamless switching between
    different reservoir types in experiments.

    Attributes:
        trained (bool): Whether the reservoir has been trained
    """

    def __init__(self):
        """Initialize the base reservoir computer."""
        self.trained = False

    @abstractmethod
    def train(self, input_data: jnp.ndarray, target_data: jnp.ndarray,
              reg_param: float = 1e-6) -> None:
        """Train the reservoir computer on the given data.

        Args:
            input_data: Input time series data of shape (time_steps, n_inputs)
            target_data: Target time series data of shape (time_steps, n_outputs)
            reg_param: Regularization parameter for ridge regression

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    @abstractmethod
    def predict(self, input_data: jnp.ndarray) -> jnp.ndarray:
        """Generate predictions for the given input data.

        Args:
            input_data: Input time series data of shape (time_steps, n_inputs)

        Returns:
            Predicted time series data of shape (time_steps, n_outputs)

        Raises:
            RuntimeError: If the reservoir hasn't been trained yet
            NotImplementedError: Must be implemented by subclasses
        """
        if not self.trained:
            raise RuntimeError("Reservoir must be trained before making predictions")

    @abstractmethod
    def get_reservoir_info(self) -> Dict[str, Any]:
        """Get information about the reservoir configuration.

        Returns:
            Dictionary containing reservoir parameters and status

        Note:
            Should include at least: n_inputs, n_outputs, trained status,
            and implementation-specific parameters
        """
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """Reset the internal state of the reservoir.

        This method should clear any internal states and set trained=False,
        effectively returning the reservoir to its initial condition.
        """
        self.trained = False

    def _validate_input_data(self, input_data: jnp.ndarray,
                           expected_features: int) -> None:
        """Validate input data dimensions and format.

        Args:
            input_data: Input data to validate
            expected_features: Expected number of input features

        Raises:
            ValueError: If input data has wrong shape or type
        """
        if not isinstance(input_data, jnp.ndarray):
            raise ValueError("Input data must be a JAX array")

        if input_data.ndim != 2:
            raise ValueError(f"Input data must be 2D, got shape {input_data.shape}")

        if input_data.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} input features, "
                f"got {input_data.shape[1]}"
            )

    def _validate_target_data(self, target_data: jnp.ndarray,
                            expected_outputs: int,
                            expected_timesteps: int) -> None:
        """Validate target data dimensions and format.

        Args:
            target_data: Target data to validate
            expected_outputs: Expected number of output features
            expected_timesteps: Expected number of time steps

        Raises:
            ValueError: If target data has wrong shape or type
        """
        if not isinstance(target_data, jnp.ndarray):
            raise ValueError("Target data must be a JAX array")

        if target_data.ndim != 2:
            raise ValueError(f"Target data must be 2D, got shape {target_data.shape}")

        if target_data.shape != (expected_timesteps, expected_outputs):
            raise ValueError(
                f"Target data shape mismatch: expected "
                f"({expected_timesteps}, {expected_outputs}), "
                f"got {target_data.shape}"
            )


class ReservoirComputerFactory:
    """Factory class for creating reservoir computer instances.

    This factory allows for easy switching between different reservoir
    implementations based on configuration parameters.
    """

    @staticmethod
    def create_reservoir(reservoir_type: str,
                        config: Union[Dict[str, Any], Any],
                        backend: str = 'cpu') -> BaseReservoirComputer:
        """Create a reservoir computer instance.

        Args:
            reservoir_type: Type of reservoir ('classical' or 'quantum')
            config: Configuration object or dictionary
            backend: Computation backend ('cpu', 'gpu', 'quantum')

        Returns:
            Configured reservoir computer instance

        Raises:
            ValueError: If unknown reservoir type is specified
            ImportError: If required dependencies are not available
        """
        if reservoir_type.lower() == 'classical':
            from .classical import ReservoirComputer
            return ReservoirComputer(config=config, backend=backend)

        elif reservoir_type.lower() == 'quantum':
            try:
                from .quantum import QuantumReservoirComputer
                return QuantumReservoirComputer(config=config, backend=backend)
            except ImportError as e:
                raise ImportError(
                    "Quantum reservoir requires PennyLane. "
                    "Install with: uv add pennylane pennylane-jax"
                ) from e

        else:
            raise ValueError(
                f"Unknown reservoir type '{reservoir_type}'. "
                "Supported types: 'classical', 'quantum'"
            )
