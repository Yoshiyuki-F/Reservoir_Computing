"""Core protocol interfaces and shared utilities for reservoir components."""

from .config_builder import build_run_config
from .interfaces import ReadoutModule, Transformer
from .presets import PresetRegistry

__all__ = ["ReadoutModule", "Transformer", "PresetRegistry", "build_run_config"]
