"""Readout modules (trainable decoders)."""

from .base import ReadoutModule
from .ridge import RidgeRegression
from .poly_ridge import PolyRidgeReadout
from .factory import ReadoutFactory

# Backwards compatibility for quantum modules expecting RidgeReadoutNumpy
RidgeReadoutNumpy = RidgeRegression

__all__ = ["ReadoutModule", "RidgeRegression", "PolyRidgeReadout", "RidgeReadoutNumpy", "ReadoutFactory"]
