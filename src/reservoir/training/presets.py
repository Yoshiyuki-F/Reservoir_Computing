from typing import Dict

from reservoir.training.config import TrainingConfig


# --- Preset Definitions ---
# The Dataclass defaults ARE the "standard".
# We only define overrides for other presets.

TRAINING_PRESETS: Dict[str, TrainingConfig] = {
    "standard": TrainingConfig(
        name = "standard",
        batch_size= 2048,
        epochs = 10,
        learning_rate = 0.001,
        seed= 0,

        # Data Splitting
        train_size=0.8,
        val_size=0.1,
        test_ratio=0.1,
    ),
}


def get_training_preset(name: str) -> TrainingConfig:
    # Training presets remain string-keyed; StrictRegistry enforces Enum keys, so use direct dict access.
    preset = TRAINING_PRESETS.get(name)
    if preset is None:
        raise KeyError(f"Training preset '{name}' not found.")
    return preset


__all__ = [
    "TrainingConfig",
    "TRAINING_PRESETS",
    "get_training_preset",
]
