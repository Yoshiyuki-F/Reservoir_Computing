"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/structs.py
Data container structures for dataset splits."""

from __future__ import annotations

from dataclasses import dataclass
from reservoir.core.types import NpF64

@dataclass
class SplitDataset:
    """Canonical dataset split container. Uses NpF64 for CPU memory storage."""

    train_X: NpF64
    train_y: NpF64
    test_X: NpF64
    test_y: NpF64
    val_X: NpF64
    val_y: NpF64
