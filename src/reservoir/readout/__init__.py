"""Readout modules (trainable decoders)."""

from .ridge import RidgeRegression

# Backwards compatibility for quantum modules expecting RidgeReadoutNumpy
RidgeReadoutNumpy = RidgeRegression

__all__ = ["RidgeRegression", "RidgeReadoutNumpy"]
