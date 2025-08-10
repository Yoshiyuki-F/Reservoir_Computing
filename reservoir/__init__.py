"""
Reservoir Computing implementation using JAX.
"""

from .reservoir_computer import ReservoirComputer
from .config import ReservoirConfig, DemoConfig, ExperimentConfig
from .data import generate_sine_data, generate_lorenz_data, generate_mackey_glass_data
from .preprocessing import normalize_data, denormalize_data
from .gpu_utils import check_gpu_available, require_gpu, print_gpu_info
from .plotting import plot_prediction_results
from .metrics import calculate_mse, calculate_mae
from . import runner

__all__ = ["ReservoirComputer", "ReservoirConfig", "ExperimentConfig", 
          "generate_sine_data", "generate_lorenz_data", "generate_mackey_glass_data",
          "normalize_data", "denormalize_data", "check_gpu_available", "require_gpu", "print_gpu_info",
          "plot_prediction_results", "calculate_mse", "calculate_mae"] 