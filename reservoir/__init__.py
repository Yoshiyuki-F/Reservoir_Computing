"""
Reservoir Computing implementation using JAX.
"""

from .reservoir_computer import ReservoirComputer
from .config import ReservoirConfig, DemoConfig, BaseConfig, create_demo_config_template
from .utils import generate_sine_data, generate_lorenz_data, generate_mackey_glass_data

__all__ = ["ReservoirComputer", "ReservoirConfig", "DemoConfig", "BaseConfig", "create_demo_config_template", "generate_sine_data", "generate_lorenz_data", "generate_mackey_glass_data"] 