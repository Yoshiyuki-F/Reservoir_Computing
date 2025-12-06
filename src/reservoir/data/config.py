#/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class BaseDatasetConfig:
    n_input: int
    n_output: int
    time_steps: int
    dt: float
    noise_level: float
    seed: int


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
class MackeyGlassConfig(BaseDatasetConfig):
    tau: int
    beta: float
    gamma: float
    n: int
    warmup_steps: int


@dataclass(frozen=True)
class MNISTConfig(BaseDatasetConfig):
    split: str
    train_fraction: float
    test_fraction: float
