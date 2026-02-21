"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/core/interfaces.py
Lightweight protocol interfaces used across preprocessing and readout components.

This module was removed unintentionally; it is restored here as the SSOT for
Transformer / Readout contracts referenced throughout the codebase.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from reservoir.core.types import JaxF64, ConfigDict, KwargsDict


@runtime_checkable
class Transformer(Protocol):
    """Stateless/fitful preprocessing contract."""

    def fit(self, features: JaxF64, y: JaxF64 | None = None) -> Transformer:
        ...

    def transform(self, features: JaxF64) -> JaxF64:
        ...

    def fit_transform(self, features: JaxF64) -> JaxF64:
        ...

    def to_dict(self) -> ConfigDict:
        ...


@runtime_checkable
class ReadoutModule(Protocol):
    """Protocol for readout components (e.g., ridge regression, FNN)."""

    def fit(self, states: JaxF64, targets: JaxF64) -> ReadoutModule:
        ...

    def predict(self, states: JaxF64) -> JaxF64:
        ...

    def to_dict(self) -> ConfigDict:
        ...


@runtime_checkable
class Adapter(Protocol):
    """Protocol for structural adapters (Step 4) that transform data and align targets."""

    def transform(self, X: JaxF64, **kwargs: KwargsDict) -> JaxF64:
        ...

    def align_targets(self, targets: JaxF64, **kwargs: KwargsDict) -> JaxF64:
        ...

    def __call__(self, X: JaxF64, **kwargs: KwargsDict) -> JaxF64:
        ...


__all__ = ["Transformer", "ReadoutModule", "Component", "Adapter"]


class Component(Protocol):
    """Minimal component contract for sequential composition."""

    def __call__(self, inputs: JaxF64, **kwargs: KwargsDict) -> JaxF64:
        ...

    def get_topology_meta(self) -> ConfigDict:
        ...
