"""Data pipelines package.

Top-level modules focus on experiment workflows (classical/quantum/FNN).
Generic helpers live under `reservoir.utils` and `reservoir.data`.
Imports here are lazy to avoid circular dependencies with models that
depend on individual pipeline modules.
"""

from .generic_runner import UniversalPipeline
from .run import run_pipeline

__all__ = [
    "UniversalPipeline",
    "run_pipeline",
]
