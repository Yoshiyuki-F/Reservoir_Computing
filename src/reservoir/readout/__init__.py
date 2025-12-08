"""Readout modules (trainable decoders)."""

from .ridge import RidgeRegression
from .factory import ReadoutFactory

# Backwards compatibility for quantum modules expecting RidgeReadoutNumpy
RidgeReadoutNumpy = RidgeRegression

__all__ = ["RidgeRegression", "RidgeReadoutNumpy", "ReadoutFactory"]
