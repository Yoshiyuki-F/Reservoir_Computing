"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/core/identifiers.py
Enum-based identifiers for experiments, datasets, and pipelines.

This module centralizes the definition of:

- Pipeline: どのモデルアーキテクチャを使うか
- Dataset: どのデータセットか（およびそれに紐づく TaskType）
- Preprocessing: 前処理のバリエーション
- ExperimentIdentifier: それらを組み合わせた「1つの実験」の識別子
"""

from __future__ import annotations

import enum

class Model(str, enum.Enum):
    """モデルアーキテクチャの種類。"""

    CLASSICAL_RESERVOIR = "classical_reservoir"
    FNN = "fnn"
    FNN_DISTILLATION = "fnn-distillation"
    RNN_DISTILLATION = "rnn-distillation"
    QUANTUM_GATE_BASED = "gate_based-quantum-reservoir"
    QUANTUM_ANALOG = "analog-quantum-reservoir"
    PASSTHROUGH = "passthrough"

    def __str__(self) -> str:
        return self.value


class Dataset(str, enum.Enum):
    """データセットの種類。Values must match DATASET_REGISTRY keys exactly."""

    SINE_WAVE = "sine_wave"
    LORENZ = "lorenz"
    MACKEY_GLASS = "mackey_glass"
    LORENZ96 = "lorenz96"
    MNIST = "mnist"

    def __str__(self) -> str:
        return self.value


class AggregationMode(str, enum.Enum):
    """State aggregation strategies for sequence-to-feature reduction."""

    LAST = "last"
    MEAN = "mean"
    LAST_MEAN = "last_mean"
    MTS = "mts"
    CONCAT = "concat"
    SEQUENCE = "sequence"

    def __str__(self) -> str:
        return self.value


class Preprocessing(str, enum.Enum):
    """前処理の種類（将来的な拡張も想定）。"""

    RAW = "raw"
    PCA = "pca"
    STANDARD_SCALER = "standard_scaler"
    DESIGN_MATRIX = "design_matrix"
    MAX_SCALER = "max_scaler"

    def __str__(self) -> str:
        return self.value

class ReadOutType(str, enum.Enum):
    """リードアウト層の種類。"""

    RidgeRegression = "ridge_regression"
    FNN = "fnn"

    def __str__(self) -> str:
        return self.value
