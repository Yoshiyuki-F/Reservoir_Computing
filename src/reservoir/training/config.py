"""
src/reservoir/training/config.py
Training configurations and Hyperparameter search spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training loops."""
    name: str

    # Training Loop
    batch_size: int
    epochs: int
    learning_rate: float
    classification: bool
    seed: int

    # Data Splitting
    train_size: float
    val_size: float
    test_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, ensuring types are JSON-safe."""
        return {
            "name": self.name,
            "batch_size": int(self.batch_size),
            "epochs": int(self.epochs),
            "learning_rate": float(self.learning_rate),
            "classification": bool(self.classification),
            "seed": int(self.seed),
            "train_size": float(self.train_size),
            "val_size": float(self.val_size),
            "test_ratio": float(self.test_ratio),
        }
