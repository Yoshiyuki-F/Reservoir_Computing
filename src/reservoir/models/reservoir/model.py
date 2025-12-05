"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/reservoir/model.py
High-level orchestration of preprocessing, reservoir dynamics, and readout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import chex
import jax.numpy as jnp

from reservoir.core.interfaces import Transformer
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
        readout_mode: str = "auto",
    ) -> None:
        self.reservoir = reservoir
        self.preprocess = preprocess
        self.readout_mode = readout_mode
        self._resolved_mode: Optional[str] = None

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

    def _get_features(self, states: jnp.ndarray, targets: Optional[jnp.ndarray]) -> jnp.ndarray:
        if states.ndim == 2:
            self._resolved_mode = self.readout_mode if self.readout_mode != "auto" else "full"
            return states

        batch, time, hidden = states.shape
        mode = self.readout_mode

        if mode == "auto":
            if targets is not None:
                t0 = targets.shape[0]
                if t0 == batch * time:
                    mode = "sequence"
                elif t0 == batch:
                    mode = "last"
                else:
                    raise ValueError(
                        f"Unable to deduce readout mode for targets {targets.shape} and states {states.shape}"
                    )
            else:
                mode = self._resolved_mode or "last"

        self._resolved_mode = mode

        if mode == "last":
            return states[:, -1, :]
        if mode == "mean":
            return jnp.mean(states, axis=1)
        if mode == "flatten":
            return states.reshape(batch, time * hidden)
        if mode == "sequence":
            return states.reshape(-1, hidden)
        raise ValueError(f"Unknown readout_mode '{mode}'")

    def predict(self, inputs: chex.Array, *, init_state: Optional[chex.Array] = None) -> chex.Array:
        prepared = self._prepare_inputs(inputs, fit=False)
        states = self._run_reservoir(prepared, init_state)
        features = self._get_features(states, None)
        return features

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
