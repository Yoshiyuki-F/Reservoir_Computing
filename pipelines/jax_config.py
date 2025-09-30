"""Common JAX configuration utilities for the Reservoir project."""

from __future__ import annotations

import threading

import jax

_CONFIG_LOCK = threading.Lock()
_CONFIGURED = False


def ensure_x64_enabled() -> None:
    """Enable 64-bit support in JAX exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    with _CONFIG_LOCK:
        if _CONFIGURED:
            return
        try:
            jax.config.update("jax_enable_x64", True)
        finally:
            _CONFIGURED = True


__all__ = ["ensure_x64_enabled"]
