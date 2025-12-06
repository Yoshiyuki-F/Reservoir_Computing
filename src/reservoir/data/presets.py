#/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/presets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from reservoir.core.presets import StrictRegistry
from .config import (
    BaseDatasetConfig,
    SineWaveConfig,
    LorenzConfig,
    MackeyGlassConfig,
    MNISTConfig,
)
from reservoir.core.identifiers import Dataset, TaskType


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    description: str
    task_type: TaskType
    config: BaseDatasetConfig
    use_dimensions: Optional[tuple[int, ...]] = None
    # V2 strict: configs are immutable; no dynamic override/merge logic.


DATASET_DEFINITIONS: Dict[Dataset, DatasetPreset] = {
    Dataset.SINE_WAVE: DatasetPreset(
        name="sine_wave",
        description="Multi-frequency sine wave composite",
        task_type=TaskType.REGRESSION,
        config=SineWaveConfig(
            n_input=1,
            n_output=1,
            time_steps=2000,
            dt=0.01,
            noise_level=0.05,
            seed=0,
            frequencies=(1.0, 2.0, 5.0),
        ),
    ),
    Dataset.LORENZ: DatasetPreset(
        name="lorenz",
        description="Lorenz attractor chaotic time series",
        task_type=TaskType.REGRESSION,
        config=LorenzConfig(
            n_input=1,
            n_output=1,
            time_steps=3000,
            dt=0.01,
            noise_level=0.01,
            seed=0,
            sigma=10.0,
            rho=28.0,
            beta=2.666667,
            warmup_steps=0,
        ),
        use_dimensions=(0,),
    ),
    Dataset.MACKEY_GLASS: DatasetPreset(
        name="mackey_glass",
        description="Mackey-Glass chaotic time series",
        task_type=TaskType.REGRESSION,
        config=MackeyGlassConfig(
            n_input=1,
            n_output=1,
            time_steps=1000,
            warmup_steps=100,
            dt=0.1,
            noise_level=0.0,
            seed=0,
            tau=17,
            beta=0.2,
            gamma=0.1,
            n=10,
        ),
    ),
    Dataset.MNIST: DatasetPreset(
        name="mnist",
        description="MNIST digit classification",
        task_type=TaskType.CLASSIFICATION,
        config=MNISTConfig(
            n_input=28,
            n_output=10,
            time_steps=28,
            dt=1.0,
            noise_level=0.0,
            seed=0,
            split="train",
            train_fraction=1.0,
            test_fraction=1.0,
        ),
    ),
}


DATASET_REGISTRY = StrictRegistry(DATASET_DEFINITIONS)
DATASET_PRESETS = DATASET_DEFINITIONS


def get_dataset_preset(dataset: Dataset) -> Optional[DatasetPreset]:
    return DATASET_REGISTRY.get(dataset)


__all__ = [
    "DATASET_REGISTRY",
    "DATASET_PRESETS",
    "DatasetPreset",
    "get_dataset_preset",
]
