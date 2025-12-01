"""
src/reservoir/models/orchestrator.py
High-level orchestration of preprocessing, reservoir dynamics, and readout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import chex
import jax.numpy as jnp

from reservoir.core.interfaces import ReservoirNode, ReadoutModule, Transformer
from reservoir.components.readout.ridge import RidgeRegression


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

    def train(
        self,
        inputs: chex.Array,
        targets: chex.Array,
        *,
        validation: Optional[tuple[chex.Array, chex.Array]] = None,
        ridge_lambdas: Optional[List[float]] = None,
        metric: str = "mse",
        init_state: Optional[chex.Array] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Fit the readout, optionally searching over ridge lambdas using a validation split."""
        # kwargs (e.g., name, batch_size) are ignored for closed-form Ridge.
        prepared = self._prepare_inputs(inputs, fit=True)
        states = self._run_reservoir(prepared, init_state)
        features = self._get_features(states, jnp.asarray(targets))

        val_features = None
        val_targets = None
        if validation is not None:
            val_X, val_y = validation
            val_prepared = self._prepare_inputs(val_X, fit=False)
            val_states = self._run_reservoir(val_prepared, init_state)
            val_features = self._get_features(val_states, jnp.asarray(val_y))
            val_targets = jnp.asarray(val_y)

        lambdas = ridge_lambdas or [self.readout.alpha if hasattr(self.readout, "alpha") else 1.0]
        best_lambda = None
        best_score: Optional[float] = None
        best_readout: Optional[RidgeRegression] = None
        search_log: Dict[float, float] = {}

        for lam in lambdas:
            candidate = RidgeRegression(alpha=float(lam), use_intercept=getattr(self.readout, "use_intercept", True))
            candidate.fit(features, targets)

            if val_features is not None and val_targets is not None:
                preds = candidate.predict(val_features)
                if metric == "accuracy":
                    pred_labels = jnp.argmax(preds, axis=-1)
                    true_labels = val_targets if val_targets.ndim == 1 else jnp.argmax(val_targets, axis=-1)
                    score_val = float(jnp.mean(pred_labels == true_labels))
                else:
                    score_val = float(jnp.mean((preds - val_targets) ** 2))
            else:
                preds = candidate.predict(features)
                if metric == "accuracy":
                    pred_labels = jnp.argmax(preds, axis=-1)
                    true_labels = targets if targets.ndim == 1 else jnp.argmax(targets, axis=-1)
                    score_val = float(jnp.mean(pred_labels == true_labels))
                else:
                    score_val = float(jnp.mean((preds - targets) ** 2))

            search_log[float(lam)] = score_val

            if best_score is None:
                best_score = score_val
                best_lambda = float(lam)
                best_readout = candidate
            else:
                if metric == "accuracy":
                    if score_val > best_score:
                        best_score = score_val
                        best_lambda = float(lam)
                        best_readout = candidate
                else:
                    if score_val < best_score:
                        best_score = score_val
                        best_lambda = float(lam)
                        best_readout = candidate

        if best_readout is not None:
            self.readout = best_readout

        return {
            "best_lambda": best_lambda,
            "validation_score": best_score,
            "search_history": search_log,
            "metric": metric,
        }

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "reservoir": self.reservoir.to_dict(),
            "readout": self.readout.to_dict(),
            "readout_mode": self.readout_mode,
        }
        if self.preprocess is not None and hasattr(self.preprocess, "to_dict"):
            data["preprocess"] = self.preprocess.to_dict()
        return data
