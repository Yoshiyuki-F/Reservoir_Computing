from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from reservoir.core.presets import PresetRegistry
from .config import DataGenerationConfig

_CONFIG_FIELDS = set(DataGenerationConfig.__dataclass_fields__.keys())


def _coerce_data_config(config: DataGenerationConfig) -> DataGenerationConfig:
    if config.time_steps is not None:
        config.time_steps = int(config.time_steps)
    if config.dt is not None:
        config.dt = float(config.dt)
    if config.noise_level is not None:
        config.noise_level = float(config.noise_level)
    if config.n_input is not None:
        config.n_input = int(config.n_input)
    if config.n_output is not None:
        config.n_output = int(config.n_output)
    if config.warmup_steps is not None:
        config.warmup_steps = int(config.warmup_steps)
    return config


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    description: str
    task_type: str
    config: DataGenerationConfig
    use_dimensions: Optional[tuple[int, ...]] = None

    def build_config(self, overrides: Optional[Mapping[str, Any]] = None) -> DataGenerationConfig:
        config_copy = copy.deepcopy(self.config)
        if not overrides:
            return _coerce_data_config(config_copy)

        params = dict(config_copy.params or {})
        nested_params = overrides.get("params") if isinstance(overrides.get("params"), Mapping) else {}

        for key, value in overrides.items():
            if key == "params":
                continue
            if key in _CONFIG_FIELDS:
                setattr(config_copy, key, value)
            else:
                params[key] = value

        if nested_params:
            params.update(nested_params)

        config_copy.params = params or None
        return _coerce_data_config(config_copy)


DATASET_DEFINITIONS: Dict[str, DatasetPreset] = {
    "sine_wave": DatasetPreset(
        name="sine_wave",
        description="Multi-frequency sine wave composite",
        task_type="regression",
        config=DataGenerationConfig(
            name="sine_wave",
            time_steps=2000,
            dt=0.01,
            noise_level=0.05,
            n_input=1,
            n_output=1,
            params={"frequencies": [1.0, 2.0, 5.0]},
        ),
    ),
    "lorenz": DatasetPreset(
        name="lorenz",
        description="Lorenz attractor chaotic time series",
        task_type="regression",
        config=DataGenerationConfig(
            name="lorenz",
            time_steps=3000,
            dt=0.01,
            noise_level=0.01,
            n_input=1,
            n_output=1,
            params={"sigma": 10.0, "rho": 28.0, "beta": 2.666667},
        ),
        use_dimensions=(0,),
    ),
    "mackey_glass": DatasetPreset(
        name="mackey_glass",
        description="Mackey-Glass chaotic time series",
        task_type="regression",
        config=DataGenerationConfig(
            name="mackey_glass",
            time_steps=1000,
            warmup_steps=100,
            dt=0.1,
            noise_level=0.0,
            n_input=1,
            n_output=1,
            params={
                "tau": 17,
                "beta": 0.2,
                "gamma": 0.1,
                "n": 10,
                "initial_value": 1.2,
            },
        ),
    ),
    "mnist": DatasetPreset(
        name="mnist",
        description="MNIST digit classification",
        task_type="classification",
        config=DataGenerationConfig(
            name="mnist",
            time_steps=28,
            dt=1.0,
            noise_level=0.0,
            n_input=28,
            n_output=10,
            params={"split": "train", "train_fraction": 1, "test_fraction": 1},
        ),
    ),
}

DATASET_ALIASES: Dict[str, str] = {
    "sine": "sine_wave",
    "sw": "sine_wave",
    "m": "mnist",
}

DATASET_REGISTRY = PresetRegistry(DATASET_DEFINITIONS, DATASET_ALIASES)

# Backwards-compatible shorthands
DATASET_PRESETS = DATASET_DEFINITIONS


def normalize_dataset_name(name: str) -> str:
    return DATASET_REGISTRY.normalize_name(name)


def get_dataset_preset(name: str) -> Optional[DatasetPreset]:
    return DATASET_REGISTRY.get(name)


__all__ = [
    "DATASET_PRESETS",
    "DATASET_REGISTRY",
    "DATASET_ALIASES",
    "DatasetPreset",
    "get_dataset_preset",
    "normalize_dataset_name",
]
