"""
src/reservoir/models/orchestrator.py
High-level orchestration of preprocessing, reservoir dynamics, and readout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import chex
import jax.numpy as jnp

from reservoir.core.interfaces import ReservoirNode, ReadoutModule, Transformer


@dataclass
class ReservoirArtifacts:
    features: jnp.ndarray
    states: jnp.ndarray


class ReservoirModel:
    """Thin orchestrator that wires preprocess, reservoir node, and readout module."""

    def __init__(
        self,
        *,
        reservoir: ReservoirNode,
        readout: ReadoutModule,
        preprocess: Optional[Transformer] = None,
        readout_mode: str = "auto",
    ) -> None:
        self.reservoir = reservoir
        self.readout = readout
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

    def fit(self, inputs: chex.Array, targets: chex.Array, *, init_state: Optional[chex.Array] = None) -> "ReservoirModel":
        prepared = self._prepare_inputs(inputs, fit=True)
        states = self._run_reservoir(prepared, init_state)
        features = self._get_features(states, jnp.asarray(targets))
        self.readout.fit(features, targets)
        return self

    def predict(self, inputs: chex.Array, *, init_state: Optional[chex.Array] = None) -> chex.Array:
        prepared = self._prepare_inputs(inputs, fit=False)
        states = self._run_reservoir(prepared, init_state)
        features = self._get_features(states, None)
        return self.readout.predict(features)

    def score(self, inputs: chex.Array, targets: chex.Array, metric: str = "mse") -> float:
        preds = self.predict(inputs)
        if preds.shape != targets.shape and preds.size == targets.size:
            preds = preds.reshape(targets.shape)
        if metric == "mse":
            return float(jnp.mean((preds - targets) ** 2))
        if metric == "accuracy":
            pred_labels = jnp.argmax(preds, axis=-1)
            true_labels = targets if targets.ndim == 1 else jnp.argmax(targets, axis=-1)
            return float(jnp.mean(pred_labels == true_labels))
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "reservoir": self.reservoir.to_dict(),
            "readout": self.readout.to_dict(),
            "readout_mode": self.readout_mode,
        }
        if self.preprocess is not None and hasattr(self.preprocess, "to_dict"):
            data["preprocess"] = self.preprocess.to_dict()
        return data
