"""
src/reservoir/models/reservoir/base.py
Base class for Reservoir Computing models implementing ReservoirNode protocol.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict

import jax.numpy as jnp


class Reservoir(ABC):
    """Abstract base class providing common scan-based trajectory generation."""

    def __init__(self, n_units: int, seed: int) -> None:
        self.n_units = n_units
        self.seed = seed

    @property
    def output_dim(self) -> int:
        return self.n_units

    @abstractmethod
    def initialize_state(self, batch_size: int = 1) -> jnp.ndarray:
        """Initialize reservoir state (must be implemented by subclasses)."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, state: jnp.ndarray, input_data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute single step dynamics returning next state and emitted features."""
        raise NotImplementedError

    def generate_trajectory(self, initial_state: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
        """Process sequences by delegating to the subclass forward implementation."""
        is_sequence_batched = inputs.ndim == 3
        inputs_batched = inputs if is_sequence_batched else inputs[None, ...]
        if initial_state.ndim == 1:
            init_batched = initial_state[None, ...]
        else:
            init_batched = initial_state if is_sequence_batched else initial_state[None, ...]

        _, outputs = self.forward(init_batched, inputs_batched)
        states = outputs.states if hasattr(outputs, "states") else outputs
        return states if is_sequence_batched else states[0]

    def to_dict(self) -> Dict[str, Any]:
        return {"n_units": self.n_units}

    def __call__(self, inputs: jnp.ndarray, **kwargs: Any) -> jnp.ndarray:
        """
        Allow reservoir nodes to be used directly in SequentialModel.
        Automatically initializes state and runs trajectory generation.
        """
        arr = jnp.asarray(inputs, dtype=jnp.float64)
        if arr.ndim not in (2, 3):
            raise ValueError(f"Reservoir input must be 2D or 3D, got {arr.shape}")
        batch_size = arr.shape[0] if arr.ndim == 3 else 1
        init_state = self.initialize_state(batch_size)
        return self.generate_trajectory(init_state, arr)
