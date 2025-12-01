"""
src/reservoir/training/presets.py
Training configurations and Hyperparameter search spaces.

V2 Architecture Compliance:
- Single Source of Truth: The dataclass defines the 'Standard' defaults.
- No Redundancy: Removed 'ridge_alpha' in favor of 'ridge_lambdas' list.
- Explicit: Config objects are complete and typed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from reservoir.core.presets import PresetRegistry


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training loops and ridge hyperparameters."""

    name: str = "standard"

    # Training Loop
    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 0.001

    # Readout Regularization (Ridge Regression)
    # Replaces 'ridge_alpha'. Defines the search space for validation.
    # Default is a log-spaced range around typical values.
    ridge_lambdas: List[float] = field(
        default_factory=lambda: [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
    )

    # Data Splitting
    train_size: float = 0.8
    val_size: float = 0.1
    test_ratio: float = 0.1

    # Task Metadata
    task_type: str = "timeseries"  # 'timeseries', 'classification', etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, ensuring types are JSON-safe."""
        return {
            "name": self.name,
            "batch_size": int(self.batch_size),
            "num_epochs": int(self.num_epochs),
            "learning_rate": float(self.learning_rate),
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
        num_epochs=1,
        ridge_lambdas=[1e-3],  # Single value for speed
    ),

    "heavy": TrainingConfig(
        name="heavy",
        batch_size=256,
        num_epochs=100,
        learning_rate=1e-4,
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