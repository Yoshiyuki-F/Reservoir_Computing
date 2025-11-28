"""/home/yoshi/PycharmProjects/Reservoir/src/core_lib/models/reservoir/base.py
Abstract base class for Reservoir Computing implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional, Sequence

import jax.numpy as jnp
from jax import Array

from core_lib.models.base import BaseModel
from core_lib.components.readout.base import BaseReadout
from core_lib.utils import ensure_x64_enabled

ensure_x64_enabled()


class BaseReservoir(ABC):
    """Abstract interface for any reservoir system (classical, quantum, etc.)."""

    def __init__(self) -> None:
        self.W_out: Optional[jnp.ndarray] = None
        self.readout_logs: Optional[Any] = None

    @property
    @abstractmethod
    def readout(self) -> BaseReadout:
        """Return the readout component used for training."""
        raise NotImplementedError

    @property
    @abstractmethod
    def washout_steps(self) -> int:
        """Number of initial timesteps to discard for regression tasks."""
        raise NotImplementedError

    @abstractmethod
    def get_states(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Run the reservoir dynamics and return raw internal states."""
        raise NotImplementedError

    @abstractmethod
    def transform_states(self, raw_states: jnp.ndarray, fit: bool = False) -> jnp.ndarray:
        """Transform raw states into a design matrix (scaling/expansion/bias)."""
        raise NotImplementedError


class BaseReservoirComputer(BaseModel, BaseReservoir):
    """Abstract base class for all reservoir computer implementations."""

    def __init__(self):
        """Initialize the base reservoir computer."""
        BaseReservoir.__init__(self)
        self.trained = False

    @abstractmethod
    def train(
        self,
        input_data: jnp.ndarray,
        target_data: jnp.ndarray,
        ridge_lambdas: Optional[Sequence[float]] = None,
    ) -> None:
        """Train the reservoir computer on the given data."""
        pass

    @abstractmethod
    def predict(self, input_data: Array) -> Array:
        """Generate predictions for the given input data."""
        self._ensure_trained()

    def evaluate(self, input_data: Array, target_data: Array) -> Dict[str, float]:
        """Compute simple regression metrics (MSE/MAE)."""
        self._ensure_trained()
        if not hasattr(self, "n_inputs") or not hasattr(self, "n_outputs"):
            raise AttributeError(
                "Subclasses must define n_inputs and n_outputs to use evaluate()."
            )

        self._validate_input_data(input_data, int(self.n_inputs))
        self._validate_target_data(
            target_data,
            expected_outputs=int(self.n_outputs),
            expected_timesteps=input_data.shape[0],
        )

        predictions = self.predict(input_data)
        if predictions.shape != target_data.shape:
            raise ValueError(
                f"Prediction shape {predictions.shape} does not match target shape {target_data.shape}"
            )

        mse = float(jnp.mean((predictions - target_data) ** 2))
        mae = float(jnp.mean(jnp.abs(predictions - target_data)))
        return {"mse": mse, "mae": mae}

    @abstractmethod
    def get_reservoir_info(self) -> Dict[str, Any]:
        """Get information about the reservoir configuration."""
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """Reset the internal state of the reservoir."""
        self.trained = False

    def _ensure_trained(self) -> None:
        """Raise if predict is called before training."""
        if not self.trained:
            raise RuntimeError("Reservoir must be trained before making predictions")

    @staticmethod
    def _validate_input_data(input_data: Array, expected_features: int) -> None:
        if not isinstance(input_data, jnp.ndarray):
            raise ValueError("Input data must be a JAX array")
        if input_data.ndim != 2:
            raise ValueError(f"Input data must be 2D, got shape {input_data.shape}")
        if input_data.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} input features, "
                f"got {input_data.shape[1]}"
            )

    @staticmethod
    def _validate_target_data(
        target_data: Array,
        expected_outputs: int,
        expected_timesteps: int,
    ) -> None:
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
    """Factory class for creating reservoir computer instances."""

    @staticmethod
    def create_reservoir(reservoir_type: str,
                        config: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
                        backend: str = 'cpu') -> BaseReservoirComputer:
        if reservoir_type.lower() == 'classical':
            from .classical import ReservoirComputer
            return ReservoirComputer(config=config, backend=backend)

        elif reservoir_type.lower() == 'quantum':
            try:
                from .quantum_gate_based import QuantumReservoirComputer
                return QuantumReservoirComputer(config=config, backend=backend)
            except ImportError as e:
                raise ImportError(
                    "Quantum reservoir requires PennyLane. "
                    "Install with: uv add pennylane pennylane-jax"
                ) from e

        elif reservoir_type.lower() in {'analog', 'quantum_analog'}:
            try:
                from .quantum_analog import AnalogQuantumReservoir
                return AnalogQuantumReservoir(config=config)
            except ImportError as e:
                raise ImportError(
                    "Analog reservoir requires QuTiP. "
                    "Install with: pip install qutip"
                ) from e

        else:
            raise ValueError(
                f"Unknown reservoir type '{reservoir_type}'. "
                "Supported types: 'classical', 'quantum_gate_based', 'quantum_analog'"
            )
