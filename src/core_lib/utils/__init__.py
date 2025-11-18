"""Utility helpers used across the core reservoir library and pipelines.

This subpackage groups together generic utilities such as:
- JAX configuration
- GPU helpers
- Metrics
- Plotting
- Preprocessing

They are re-exported here for convenience, so callers can import from
`core_lib.utils.*` or from `core_lib.utils` directly.
"""

from .jax_config import ensure_x64_enabled  # noqa: F401
from .gpu_utils import (  # noqa: F401
    check_gpu_available,
    require_gpu,
    print_gpu_info,
)
from .metrics import calculate_mse, calculate_mae  # noqa: F401
from .plotting import (  # noqa: F401
    plot_prediction_results,
    plot_classification_results,
    plot_epoch_metric,
)
from .preprocessing import (  # noqa: F401
    normalize_data,
    denormalize_data,
    create_sliding_windows,
)

__all__ = [
    "ensure_x64_enabled",
    "check_gpu_available",
    "require_gpu",
    "print_gpu_info",
    "calculate_mse",
    "calculate_mae",
    "plot_prediction_results",
    "plot_classification_results",
    "plot_epoch_metric",
    "normalize_data",
    "denormalize_data",
    "create_sliding_windows",
]
