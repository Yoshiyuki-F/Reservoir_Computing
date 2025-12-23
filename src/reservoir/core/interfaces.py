"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/core/interfaces.py
Lightweight protocol interfaces used across preprocessing and readout components.

This module was removed unintentionally; it is restored here as the SSOT for
Transformer / Readout contracts referenced throughout the codebase.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable
import jax.numpy as jnp

from reservoir.core.identifiers import TaskType


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
    """Protocol for readout components (e.g., ridge regression, FNN)."""

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
        *,
        task_type: TaskType = TaskType.REGRESSION,
    ) -> tuple[float, dict[float, float], dict[float, float]]:
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
