"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/identifiers.py
Dataset identifiers.
"""
from __future__ import annotations

import enum


class Dataset(enum.StrEnum):
    """Available dataset identifiers."""

    SINE_WAVE = "sine_wave"
    LORENZ = "lorenz"
    LORENZ96 = "lorenz96"
    MACKEY_GLASS = "mackey_glass"
    MNIST = "mnist"

    def __str__(self) -> str:
        return self.value

