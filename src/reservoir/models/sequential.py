"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/sequential.py
Generic sequential container used for composing preprocessing, reservoir, and aggregators."""
from __future__ import annotations

from typing import Any, Iterable, List, Optional, Dict

import jax.numpy as jnp


class SequentialModel:
    """Minimal sequential container that applies a list of callable layers."""

    def __init__(self, layers: Iterable[Any]):
        self.layers: List[Any] = list(layers)
        self.topology_meta: Dict[str, Any] = {}
        # Expose common subcomponents for compatibility with existing consumers.
        self.preprocess = None
        self.reservoir = None
        self.aggregator = None
        for layer in self.layers:
            if self.preprocess is None and hasattr(layer, "transform"):
                self.preprocess = layer
            if self.reservoir is None and hasattr(layer, "reservoir"):
                self.reservoir = getattr(layer, "reservoir")
            if self.aggregator is None and layer.__class__.__name__.lower().find("aggregator") != -1:
                self.aggregator = layer

    def _apply_layer(self, layer: Any, inputs: jnp.ndarray) -> jnp.ndarray:
        if hasattr(layer, "__call__"):
            try:
                return layer(inputs)
            except TypeError:
                try:
                    return layer(inputs, None)
                except TypeError:
                    pass
        if hasattr(layer, "predict"):
            return layer.predict(inputs)
        if hasattr(layer, "transform"):
            return layer.transform(inputs)
        raise TypeError(f"Layer {layer} is not callable.")

    def __call__(self, inputs: jnp.ndarray, **kwargs: Any) -> jnp.ndarray:
        data = inputs
        for layer in self.layers:
            data = self._apply_layer(layer, data)
        return data

    def predict(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return self.__call__(inputs)

    def train(self, inputs: jnp.ndarray, targets: Optional[jnp.ndarray] = None, **kwargs: Any) -> Dict[str, Any]:
        data = inputs
        logs: Dict[str, Any] = {}
        for layer in self.layers:
            if hasattr(layer, "fit_transform"):
                data = layer.fit_transform(data)
                continue
            if hasattr(layer, "train"):
                try:
                    logs = layer.train(data, targets, **kwargs) or {}
                except TypeError:
                    logs = layer.train(data, targets) or {}
                # After a trainable layer, continue data flow for any downstream layers.
                if hasattr(layer, "predict"):
                    data = layer.predict(data)
                else:
                    data = self._apply_layer(layer, data)
            else:
                data = self._apply_layer(layer, data)
        return logs

    def get_topology_meta(self) -> Dict[str, Any]:
        return getattr(self, "topology_meta", {}) or {}
