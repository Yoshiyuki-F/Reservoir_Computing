"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/reservoir/model.py
High-level orchestration of preprocessing, reservoir dynamics, and readout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import chex
import jax.numpy as jnp

from reservoir.core.interfaces import Transformer
from reservoir.components.preprocess.aggregator import StateAggregator
from reservoir.models.reservoir.base import Reservoir


@dataclass
class ReservoirArtifacts:
    features: jnp.ndarray
    states: jnp.ndarray


class ReservoirModel:
    """Orchestrator that wires preprocess and reservoir node. Returns features; readout is external."""

    def __init__(
        self,
        *,
        reservoir: Reservoir,
        preprocess: Optional[Transformer] = None,
        readout_mode: str = "last",
    ) -> None:
        self.reservoir = reservoir
        self.preprocess = preprocess
        self.aggregator = StateAggregator(mode=readout_mode)

    def _prepare_inputs(self, inputs: jnp.ndarray, *, fit: bool) -> jnp.ndarray:
        arr = jnp.asarray(inputs, dtype=jnp.float64)
        if self.preprocess is None:
            return arr
        return self.preprocess.fit_transform(arr) if fit else self.preprocess.transform(arr)

    def _run_reservoir(self, inputs: jnp.ndarray, init_state: Optional[jnp.ndarray]) -> jnp.ndarray:
        if init_state is None:
            batch_size = inputs.shape[0] if inputs.ndim == 3 else 1
            init_state = self.reservoir.initialize_state(batch_size)
        return self.reservoir.generate_trajectory(init_state, inputs)

    def predict(self, inputs: chex.Array, *, init_state: Optional[chex.Array] = None) -> chex.Array:
        prepared = self._prepare_inputs(inputs, fit=False)
        states = self._run_reservoir(prepared, init_state)
        return self.aggregator.transform(states)

    def __call__(self, inputs: chex.Array, *, init_state: Optional[chex.Array] = None) -> chex.Array:
        return self.predict(inputs, init_state=init_state)

    def train(
        self,
        inputs: chex.Array,
        targets: chex.Array,
        *,
        init_state: Optional[chex.Array] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Reservoir model has no internal training; runner handles readout and evaluation."""
        # _ = (inputs, targets, init_state, kwargs)
        return {}
