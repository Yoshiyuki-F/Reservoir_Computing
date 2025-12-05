"""Core protocol interfaces and shared utilities for reservoir components."""

from .config_builder import build_run_config
from .presets import PresetRegistry
from .interfaces import Transformer, ReadoutModule

__all__ = ["PresetRegistry", "build_run_config", "Transformer", "ReadoutModule"]
