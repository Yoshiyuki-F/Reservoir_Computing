from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from reservoir.core.presets import PresetRegistry


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training loops and ridge hyperparameters."""

    name: str = "standard"
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-3
    ridge_alpha: float = 1e-3
    ridge_lambdas: List[float] = field(default_factory=lambda: [1e-3])
    train_size: float = 0.8
    val_size: float = 0.1
    test_ratio: float = 0.1
    task_type: str = "timeseries"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "batch_size": int(self.batch_size),
            "num_epochs": int(self.num_epochs),
            "learning_rate": float(self.learning_rate),
            "ridge_alpha": float(self.ridge_alpha),
            "ridge_lambdas": [float(v) for v in self.ridge_lambdas],
            "train_size": float(self.train_size),
            "val_size": float(self.val_size),
            "test_ratio": float(self.test_ratio),
            "task_type": self.task_type,
        }


TRAINING_DEFINITIONS: Dict[str, TrainingConfig] = {
    "standard": TrainingConfig(
        name="standard",
        batch_size=128,
        num_epochs=20,
        learning_rate=0.001,
        ridge_alpha=1e-3,
        ridge_lambdas=[-7, 7, 15],
        train_size=0.8,
        val_size=0.1,
        test_ratio=0.1,
        task_type="timeseries",
    ),
    "quick_test": TrainingConfig(
        name="quick_test",
        batch_size=32,
        num_epochs=1,
        learning_rate=1e-3,
        ridge_alpha=1e-3,
        ridge_lambdas=[1e-3],
        train_size=0.8,
        val_size=0.1,
        test_ratio=0.1,
        task_type="timeseries",
    ),
    "heavy": TrainingConfig(
        name="heavy",
        batch_size=256,
        num_epochs=100,
        learning_rate=1e-4,
        ridge_alpha=1e-5,
        ridge_lambdas=[1e-5, 1e-4, 1e-3],
        train_size=0.8,
        val_size=0.1,
        test_ratio=0.1,
        task_type="timeseries",
    ),
}

TRAINING_ALIASES: Dict[str, str] = {
    "std": "standard",
    "quick": "quick_test",
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
