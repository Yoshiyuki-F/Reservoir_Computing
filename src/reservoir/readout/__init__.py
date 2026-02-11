"""Readout modules (trainable decoders)."""

from .ridge import RidgeRegression
from .poly_ridge import PolyRidgeReadout
from .factory import ReadoutFactory

# Backwards compatibility for quantum modules expecting RidgeReadoutNumpy
RidgeReadoutNumpy = RidgeRegression

__all__ = ["RidgeRegression", "PolyRidgeReadout", "RidgeReadoutNumpy", "ReadoutFactory"]
