"""Utility helpers used across the core reservoir library and pipelines.

This subpackage groups together generic utilities such as:
- JAX configuration
- GPU helpers
- Metrics

They are re-exported here for convenience, so callers can import from
`reservoir.utils.*` or from `reservoir.utils` directly.
"""

from .jax_config import ensure_x64_enabled  # noqa: F401
from .gpu_utils import (  # noqa: F401
    check_gpu_available,
    require_gpu,
    print_gpu_info,
)
from .metrics import compute_score, calculate_chaos_metrics  # noqa: F401

__all__ = [
    "ensure_x64_enabled",
    "check_gpu_available",
    "require_gpu",
    "print_gpu_info",
    "compute_score",
    "calculate_chaos_metrics",
]
