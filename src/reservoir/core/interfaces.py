"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/core/interfaces.py
Lightweight protocol interfaces used across preprocessing and readout components.

This module was removed unintentionally; it is restored here as the SSOT for
Transformer / Readout contracts referenced throughout the codebase.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable, Sequence, Tuple
import jax.numpy as jnp


@runtime_checkable
class Transformer(Protocol):
    """Stateless/fitful preprocessing contract."""

    def fit(self, features: Any, y: Any = None) -> "Transformer":
        ...

    def transform(self, features: Any) -> Any:
        ...

    def fit_transform(self, features: Any) -> Any:
        ...

    def to_dict(self) -> Dict[str, Any]:
        ...


@runtime_checkable
class ReadoutModule(Protocol):
    """Protocol for readout components (e.g., ridge regression)."""

    ridge_lambda: float
    coef_: Optional[jnp.ndarray]

    def fit(self, states: Any, targets: Any) -> Any:
        ...

    def predict(self, states: Any) -> Any:
        ...

    def fit_and_search(
        self,
        train_states: Any,
        train_targets: Any,
        val_states: Any,
        val_targets: Any,
        lambdas: Sequence[float] | Any,
        *,
        metric: str = "mse",
    ) -> Tuple[float, Dict[float, float], Dict[float, float]]:
        ...

    def to_dict(self) -> Dict[str, Any]:
        ...


__all__ = ["Transformer", "ReadoutModule", "Component"]


class Component(Protocol):
    """Minimal component contract for sequential composition."""

    def __call__(self, inputs: jnp.ndarray, **kwargs: Any) -> jnp.ndarray:
        ...

    def get_topology_meta(self) -> Dict[str, Any]:
        ...
