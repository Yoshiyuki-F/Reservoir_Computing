#/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from reservoir.core.identifiers import TaskType

@dataclass(frozen=True)
class BaseDatasetConfig:
    n_input: int
    n_output: int
    time_steps: int
    dt: float
    noise_level: float
    seed: Optional[int]

@dataclass(frozen=True)
class SineWaveConfig(BaseDatasetConfig):
    frequencies: Tuple[float, ...]

@dataclass(frozen=True)
class LorenzConfig(BaseDatasetConfig):
    sigma: float
    rho: float
    beta: float
    warmup_steps: int

@dataclass(frozen=True)
class Lorenz96Config(BaseDatasetConfig):
    F: float
    warmup_steps: int

@dataclass(frozen=True)
class MackeyGlassConfig(BaseDatasetConfig):
    tau: int
    beta: float
    gamma: float
    n: int
    warmup_steps: int
    downsample: int

@dataclass(frozen=True)
class MNISTConfig(BaseDatasetConfig):
    split: str
    train_fraction: float
    test_fraction: float

@dataclass(frozen=True)
class DatasetPreset:
    name: str
    description: str
    task_type: TaskType
    config: BaseDatasetConfig
    use_dimensions: Optional[tuple[int, ...]]