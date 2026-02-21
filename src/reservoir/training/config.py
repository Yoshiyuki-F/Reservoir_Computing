"""
src/reservoir/training/config.py
Training configurations and Hyperparameter search spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reservoir.core.types import ConfigDict


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training loops."""
    name: str

    # Training Loop
    batch_size: int
    epochs: int
    learning_rate: float
    seed: int

    # Learning Rate Scheduler
    scheduler_type: str | None  # "cosine", "piecewise", or None (constant)
    warmup_epochs: int

    # Data Splitting
    train_size: float
    val_size: float
    test_ratio: float

    def to_dict(self) -> ConfigDict:
        """Convert to dictionary, ensuring types are JSON-safe."""
        return {
            "name": self.name,
            "batch_size": int(self.batch_size),
            "epochs": int(self.epochs),
            "learning_rate": float(self.learning_rate),
            "scheduler_type": self.scheduler_type,
            "warmup_epochs": int(self.warmup_epochs),
            "seed": int(self.seed),
            "train_size": float(self.train_size),
            "val_size": float(self.val_size),
            "test_ratio": float(self.test_ratio),
        }
