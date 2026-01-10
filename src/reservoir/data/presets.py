#/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/presets.py
from __future__ import annotations

from typing import Dict, Optional

from reservoir.core.presets import StrictRegistry
from .config import (
    SineWaveConfig,
    LorenzConfig,
    Lorenz96Config,
    MackeyGlassConfig,
    MNISTConfig, DatasetPreset,
)
from reservoir.core.identifiers import Dataset



DATASET_DEFINITIONS: Dict[Dataset, DatasetPreset] = {
    # Dataset.SINE_WAVE: DatasetPreset(
    #     name="sine_wave",
    #     description="Multi-frequency sine wave composite",
    #     classification=False,
    #     config=SineWaveConfig(
    #         n_input=1,
    #         n_output=1,
    #         time_steps=2000,
    #         noise_level=0.05,
    #         seed=0,
    #         frequencies=(1.0, 2.0, 5.0),
    #     ),
    #     use_dimensions=(0,),
    # ),
    Dataset.LORENZ: DatasetPreset(
        name="lorenz",
        description="Lorenz attractor chaotic time series",
        classification=False,
        config=LorenzConfig(
            n_input=1,
            n_output=1,
            dt=0.01,
            noise_level=0.01,
            seed=0,
            sigma=10.0,
            rho=28.0,
            beta=2.666667,
            lyapunov_time_unit=1.1,  # 1 LT ≈ 1.1 time units @ dt=0.01 → 110 steps
            washup_lt=5,
            train_lt=60,
            val_lt=5,
            test_lt=10,
        ),
        use_dimensions=(0,),
    ),
    # Dataset.LORENZ96: DatasetPreset(
    #     name="lorenz96",
    #     description="Lorenz 96 chaotic system",
    #     classification=False,
    #     config=Lorenz96Config(
    #         n_input=40,
    #         n_output=40,
    #         dt=0.05,
    #         noise_level=0.0,
    #         seed=0,
    #         F=8.0,
    #         washup_lt=5,
    #         lyapunov_time_unit=0,  # TODO: calculate correct value
    #     ),
    #     use_dimensions=None,
    # ),
    Dataset.MACKEY_GLASS: DatasetPreset(
        name="mackey_glass",
        description="Mackey-Glass chaotic time series",
        classification=False,
        config=MackeyGlassConfig(
            n_input=1,
            n_output=1,
            dt=1,
            noise_level=0.0,
            seed=0,
            tau=17,
            beta=0.2,
            gamma=0.1,
            n=10,
            downsample=1,
            lyapunov_time_unit=166.6,  # Mackey-Glass LT
            washup_lt=5,
            train_lt=40,
            val_lt=5,
            test_lt=10,
        ),
        use_dimensions=None
    ),
    Dataset.MNIST: DatasetPreset(
        name="mnist",
        description="MNIST digit classification",
        classification=True,
        config=MNISTConfig(
            n_input=28,
            n_output=10,
            time_steps=28,
            noise_level=0.0,
            seed=0,
            split="train",
            train_fraction=1.0,
            test_fraction=1.0,
        ),
        use_dimensions=None
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
