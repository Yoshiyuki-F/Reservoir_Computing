"""
Reservoir Computing implementation using JAX.
"""

from .reservoir_computer import ReservoirComputer
from .config import ReservoirConfig, DemoConfig, ExperimentConfig, DataGenerationConfig
from .data import generate_sine_data, generate_lorenz_data, generate_mackey_glass_data
from utils.preprocessing import normalize_data, denormalize_data
from utils.gpu_utils import check_gpu_available, require_gpu, print_gpu_info
from utils.plotting import plot_prediction_results
from utils.metrics import calculate_mse, calculate_mae
from . import runner

__all__ = ["ReservoirComputer", "ReservoirConfig", "ExperimentConfig", "DataGenerationConfig", 
          "generate_sine_data", "generate_lorenz_data", "generate_mackey_glass_data",
          "normalize_data", "denormalize_data", "check_gpu_available", "require_gpu", "print_gpu_info",
          "plot_prediction_results", "calculate_mse", "calculate_mae"] 