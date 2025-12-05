"""Reservoir model implementations and shared abstract base class."""

from .base import Reservoir
from .classical import ClassicalReservoir

__all__ = ["Reservoir", "ClassicalReservoir"]
