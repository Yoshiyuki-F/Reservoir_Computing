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

class Model(enum.StrEnum):
    """モデルアーキテクチャの種類。"""

    CLASSICAL_RESERVOIR = "classical_reservoir"
    FNN = "fnn"
    FNN_DISTILLATION = "fnn_distillation"
    RNN_DISTILLATION = "rnn-distillation"
    QUANTUM_RESERVOIR = "quantum_reservoir"
    PASSTHROUGH = "passthrough"

    def __str__(self) -> str:
        return self.value


class Dataset(enum.StrEnum):
    """データセットの種類。Values must match DATASET_REGISTRY keys exactly."""

    SINE_WAVE = "sine_wave"
    LORENZ = "lorenz"
    MACKEY_GLASS = "mackey_glass"
    LORENZ96 = "lorenz96"
    MNIST = "mnist"

    def __str__(self) -> str:
        return self.value


class AggregationMode(enum.StrEnum):
    """State aggregation strategies for sequence-to-feature reduction."""

    LAST = "last"
    MEAN = "mean"
    LAST_MEAN = "last_mean"
    MTS = "mts"
    CONCAT = "concat"
    SEQUENCE = "sequence"

    def __str__(self) -> str:
        return self.value


class Preprocessing(enum.StrEnum):
    """前処理の種類（将来的な拡張も想定）。"""

    RAW = "raw"
    PCA = "pca"
    STANDARD_SCALER = "standard_scaler"
    DESIGN_MATRIX = "design_matrix"
    CUSTOM_RANGE_SCALER = "custom_range_scaler"

    def __str__(self) -> str:
        return self.value

class ReadOutType(enum.StrEnum):
    """リードアウト層の種類。"""

    RidgeRegression = "ridge_regression"
    FNN = "fnn"

    def __str__(self) -> str:
        return self.value
