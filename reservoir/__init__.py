"""
Reservoir Computing implementation using JAX.
"""

from .reservoir_computer import ReservoirComputer
from .utils import generate_sine_data, generate_lorenz_data, generate_mackey_glass_data

__all__ = ["ReservoirComputer", "generate_sine_data", "generate_lorenz_data", "generate_mackey_glass_data"] 