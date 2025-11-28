"""Reservoir model implementations."""

from .base_reservoir import BaseReservoirComputer, ReservoirComputerFactory
from .classical import ReservoirComputer
from .factory import ReservoirFactory
try:
    from .gate_based_quantum import QuantumReservoirComputer
    QUANTUM_AVAILABLE = True
except ImportError:
    QuantumReservoirComputer = None
    QUANTUM_AVAILABLE = False

try:
    from .analog_quantum import AnalogQuantumReservoir
    ANALOG_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    AnalogQuantumReservoir = None
    ANALOG_AVAILABLE = False

from core_lib.core import (
    ExperimentConfig,
    DataGenerationConfig,
    TrainingConfig,
    DemoConfig,
)

from .config import (
    ReservoirConfig,
    QuantumReservoirConfig,
    AnalogQuantumReservoirConfig,
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
    "ReservoirFactory",
    "ReservoirComputer",
    "QuantumReservoirComputer",
    "AnalogQuantumReservoir",
    "QUANTUM_AVAILABLE",
    "ANALOG_AVAILABLE",
    "ExperimentConfig",
    "ReservoirConfig",
    "QuantumReservoirConfig",
    "AnalogQuantumReservoirConfig",
    "DataGenerationConfig",
    "TrainingConfig",
    "DemoConfig",
    # Optional experiment utilities
    *(["run_experiment"] if run_experiment else []),
]
