"""
src/reservoir/training/config.py
Training configurations and Hyperparameter search spaces.

V2 Architecture Compliance:
- Single Source of Truth: The dataclass defines the defaults (including ridge_lambda).
- Explicit: Config objects are complete, typed, and validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training loops and ridge hyperparameters."""
    name: str

    # Training Loop
    batch_size: int
    epochs: int
    learning_rate: float
    classification: bool
    seed: int

    # Readout Regularization (Ridge Regression)
    # 'ridge_lambda' is the canonical default regularization strength when no search is run.
    ridge_lambda: float # no use but needed for type consistency
    # Defines the search space for validation. Defaults to a log-spaced range around typical values.
    ridge_lambdas: List[float]

    # Data Splitting //TODO test is already defined at MNIST so what gives test_ratio?
    train_size: float
    val_size: float
    test_ratio: float

    def __post_init__(self) -> None:
        if self.ridge_lambda is None or float(self.ridge_lambda) <= 0.0:
            raise ValueError("TrainingConfig.ridge_lambda must be a positive value defined in the preset.")
        if not self.ridge_lambdas:
            raise ValueError("TrainingConfig.ridge_lambdas must be a non-empty sequence.")
        if any(float(lam) <= 0.0 for lam in self.ridge_lambdas):
            raise ValueError("TrainingConfig.ridge_lambdas must contain only positive values.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, ensuring types are JSON-safe."""
        return {
            "name": self.name,
            "batch_size": int(self.batch_size),
            "epochs": int(self.epochs),
            "learning_rate": float(self.learning_rate),
            "classification": bool(self.classification),
            "seed": int(self.seed),
            "ridge_lambda": float(self.ridge_lambda),
            "ridge_lambdas": [float(v) for v in self.ridge_lambdas],
            "train_size": float(self.train_size),
            "val_size": float(self.val_size),
            "test_ratio": float(self.test_ratio),
        }
