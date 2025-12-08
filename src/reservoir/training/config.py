"""
src/reservoir/training/config.py
Training configurations and Hyperparameter search spaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training loops and hyperparameter search."""
    name: str

    # Training Loop
    batch_size: int
    epochs: int
    learning_rate: float
    classification: bool
    seed: int

    # Readout Regularization search space (used by Ridge or similar)
    ridge_lambdas: List[float]

    # Data Splitting //TODO test is already defined at MNIST so what gives test_ratio?
    train_size: float
    val_size: float
    test_ratio: float

    def __post_init__(self) -> None:
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
            "ridge_lambdas": [float(v) for v in self.ridge_lambdas],
            "train_size": float(self.train_size),
            "val_size": float(self.val_size),
            "test_ratio": float(self.test_ratio),
        }
