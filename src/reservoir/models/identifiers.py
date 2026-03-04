"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/identifiers.py
Model architecture identifiers.
"""
from __future__ import annotations

import enum


class Model(enum.StrEnum):
    """モデルアーキテクチャの種類。"""

    CLASSICAL_RESERVOIR = "classical_reservoir"
    FNN = "fnn"
    FNN_DISTILLATION_CLASSICAL = "fnn_distillation_classical"
    FNN_DISTILLATION_QUANTUM = "fnn_distillation_quantum"
    RNN_DISTILLATION = "rnn-distillation"
    QUANTUM_RESERVOIR = "quantum_reservoir"
    PASSTHROUGH = "passthrough"

    def __str__(self) -> str:
        return self.value

