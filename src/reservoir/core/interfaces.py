"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/core/interfaces.py
Lightweight protocol interfaces used across preprocessing and readout components.

This module was removed unintentionally; it is restored here as the SSOT for
Transformer / Readout contracts referenced throughout the codebase.
"""
from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable
import jax.numpy as jnp
from reservoir.core.types import JaxF64


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

    def to_dict(self) -> Dict[str, Any]:
        ...


__all__ = ["Transformer", "ReadoutModule", "Component"]


class Component(Protocol):
    """Minimal component contract for sequential composition."""

    def __call__(self, inputs: JaxF64, **kwargs: Any) -> JaxF64:
        ...

    def get_topology_meta(self) -> Dict[str, Any]:
        ...
