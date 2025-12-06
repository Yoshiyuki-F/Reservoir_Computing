"""Core protocol interfaces and shared utilities for reservoir components."""

from .presets import StrictRegistry
from .interfaces import Transformer, ReadoutModule

__all__ = ["StrictRegistry", "Transformer", "ReadoutModule"]
