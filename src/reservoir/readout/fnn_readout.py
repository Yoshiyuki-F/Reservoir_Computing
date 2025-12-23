"""FNN-based readout module implementing ReadoutModule protocol."""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import jax.numpy as jnp

from reservoir.core.interfaces import ReadoutModule
from reservoir.models.config import FNNConfig
from reservoir.models.nn.fnn import FNNModel
from reservoir.training.config import TrainingConfig


class FNNReadout(ReadoutModule):
    """FNN-based readout using FNNModel as backend."""

    def __init__(
        self,
        hidden_layers: Optional[Tuple[int, ...]] = None,
        training_config: Optional[TrainingConfig] = None
    ) -> None:
        self.hidden_layers = hidden_layers or ()
        self.training_config = training_config
        self._model: Optional[FNNModel] = None
        self._input_dim: Optional[int] = None
        self._output_dim: Optional[int] = None

    def fit(self, states: jnp.ndarray, targets: jnp.ndarray) -> "FNNReadout":
        """Fit the FNN readout on states and targets."""
        X = jnp.asarray(states, dtype=jnp.float32)
        y = jnp.asarray(targets, dtype=jnp.float32)

        if X.ndim != 2:
            raise ValueError(f"States must be 2D, got {X.shape}")
        if y.ndim == 1:
            y = y[:, None]

        self._input_dim = X.shape[1]
        self._output_dim = y.shape[1]

        # Create FNNModel with appropriate dimensions
        fnn_config = FNNConfig(hidden_layers=self.hidden_layers)
        if self.training_config is None:
            raise ValueError("FNNReadout requires training_config. Use Factory to create properly configured instance.")
        self._model = FNNModel(
            model_config=fnn_config,
            training_config=self.training_config,
            input_dim=self._input_dim,
            output_dim=self._output_dim
        )

        # Train the model
        self._model.train(X, y)
        return self

    def predict(self, states: jnp.ndarray) -> jnp.ndarray:
        """Predict using the trained FNN."""
        if self._model is None:
            raise RuntimeError("FNNReadout is not fitted yet.")
        X = jnp.asarray(states, dtype=jnp.float32)
        return self._model.predict(X)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the FNN readout."""
        return {
            "hidden_layers": self.hidden_layers,
            "input_dim": self._input_dim,
            "output_dim": self._output_dim
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FNNReadout":
        """Deserialize an FNN readout (note: model weights not preserved)."""
        return cls(hidden_layers=tuple(data.get("hidden_layers", ())))
