"""Reservoir model implementations."""

from .base_reservoir import BaseReservoirComputer, ReservoirComputerFactory
from .classical import ReservoirComputer
try:
    from .gatebased_quantum import QuantumReservoirComputer
    QUANTUM_AVAILABLE = True
except ImportError:
    QuantumReservoirComputer = None
    QUANTUM_AVAILABLE = False

from configs.core import (
    ExperimentConfig,
    DataGenerationConfig,
    TrainingConfig,
    DemoConfig,
)

from .config import (
    ReservoirConfig,
    QuantumReservoirConfig,
)

# Data preparation is now model-agnostic and in pipelines

try:
    from .__main__ import run_experiment
except ImportError:
    run_experiment = None

# QuantumReservoirComputer is already handled above

__all__ = [
    "BaseReservoirComputer",
    "ReservoirComputerFactory",
    "ReservoirComputer",
    "QuantumReservoirComputer",
    "QUANTUM_AVAILABLE",
    "ExperimentConfig",
    "ReservoirConfig",
    "QuantumReservoirConfig",
    "DataGenerationConfig",
    "TrainingConfig",
    "DemoConfig",
    # Optional experiment utilities
    *(["run_experiment"] if run_experiment else []),
]
