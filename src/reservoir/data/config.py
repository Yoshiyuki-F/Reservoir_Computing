#/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class BaseDatasetConfig:
    n_input: int
    n_output: int
    noise_level: float
    seed: Optional[int]

@dataclass(frozen=True)
class MNISTConfig(BaseDatasetConfig):
    split: str
    train_fraction: float
    test_fraction: float
    time_steps: int

@dataclass(frozen=True)
class SineWaveConfig(BaseDatasetConfig):
    frequencies: Tuple[float, ...]
    time_steps: int

@dataclass(frozen=True)
class ChaosDatasetConfig(BaseDatasetConfig):
    lyapunov_time_unit: float   # Steps per Lyapunov time (e.g., 110 for dt=0.01) = int(LT/dt)
    washup_lt: int
    train_lt: int
    val_lt: int
    test_lt: int
    dt: float

@dataclass(frozen=True)
class LorenzConfig(ChaosDatasetConfig):
    sigma: float
    rho: float
    beta: float


@dataclass(frozen=True)
class Lorenz96Config(ChaosDatasetConfig):
    F: float


@dataclass(frozen=True)
class MackeyGlassConfig(ChaosDatasetConfig):
    tau: int
    beta: float
    gamma: float
    n: int
    downsample: int


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    description: str
    classification: bool
    config: BaseDatasetConfig
    use_dimensions: Optional[tuple[int, ...]]