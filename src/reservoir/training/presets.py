"""
src/reservoir/training/presets.py
Training configurations and Hyperparameter search spaces.

V2 Architecture Compliance:
- Single Source of Truth: The dataclass defines the defaults (including ridge_lambda).
- Explicit: Config objects are complete, typed, and validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np
from reservoir.core.presets import PresetRegistry


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training loops and ridge hyperparameters."""

    name: str = "standard"

    # Training Loop
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 0.001
    classification: bool = False
    seed: int = 0

    # Readout Regularization (Ridge Regression)
    # 'ridge_lambda' is the canonical default regularization strength when no search is run.
    ridge_lambda: float = 1e-7 # no use but needed for type consistency
    # Defines the search space for validation. Defaults to a log-spaced range around typical values.
    ridge_lambdas: List[float] = field(
        default_factory=lambda: np.logspace(-15, -5, 30).tolist()
    )

    # Data Splitting //TODO test is already defined at MNIST so what gives test_ratio?
    train_size: float = 0.8
    val_size: float = 0.1
    test_ratio: float = 0.1

    # Task Metadata
    task_type: str = "timeseries"  # 'timeseries', 'classification', etc.

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
            "task_type": self.task_type,
        }


# --- Preset Definitions ---
# The Dataclass defaults ARE the "standard".
# We only define overrides for other presets.

TRAINING_DEFINITIONS: Dict[str, TrainingConfig] = {
    "standard": TrainingConfig(),  # Uses all defaults from above

    "quick_test": TrainingConfig(
        name="quick_test",
        batch_size=32,
        epochs=1,
        ridge_lambda=1e-3,
        ridge_lambdas=[1e-3],  # Single value for speed
    ),

    "heavy": TrainingConfig(
        name="heavy",
        batch_size=256,
        epochs=100,
        learning_rate=1e-4,
        ridge_lambda=1e-2,
        # Finer-grained search space
        ridge_lambdas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    ),
}

TRAINING_ALIASES: Dict[str, str] = {
    "std": "standard",
    "quick": "quick_test",
    "debug": "quick_test",
}

TRAINING_REGISTRY = PresetRegistry(TRAINING_DEFINITIONS, TRAINING_ALIASES)
TRAINING_PRESETS = TRAINING_DEFINITIONS


def normalize_training_name(name: str) -> str:
    return TRAINING_REGISTRY.normalize_name(name)


def get_training_preset(name: str) -> TrainingConfig:
    return TRAINING_REGISTRY.get_or_default(name, "standard")


__all__ = [
    "TrainingConfig",
    "TRAINING_PRESETS",
    "TRAINING_REGISTRY",
    "TRAINING_ALIASES",
    "get_training_preset",
    "normalize_training_name",
]
