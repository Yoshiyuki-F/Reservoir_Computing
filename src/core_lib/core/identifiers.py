"""Enum-based identifiers for experiments, datasets, and pipelines.

This module centralizes the definition of:

- Pipeline: どのモデルアーキテクチャを使うか
- Dataset: どのデータセットか（およびそれに紐づく TaskType）
- Preprocessing: 前処理のバリエーション
- ExperimentIdentifier: それらを組み合わせた「1つの実験」の識別子
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass
from typing import List


class Pipeline(enum.Enum):
    """実験パイプライン（モデルアーキテクチャ）の種類。"""

    CLASSICAL_RESERVOIR = "classical"
    FNN = "fnn"
    FNN_B_DASH = "fnn-b-dash"
    GATEBASED_QUANTUM = "gatebased-quantum"
    ANALOG_QUANTUM = "analog-quantum"

    def __str__(self) -> str:
        return self.value


class TaskType(enum.Enum):
    """タスクの種類。"""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"

    def __str__(self) -> str:
        return self.value


class Dataset(enum.Enum):
    """データセットの種類。"""

    SINE_WAVE = "sine_wave"
    LORENZ = "lorenz"
    MACKEY_GLASS = "mackey_glass"
    MNIST = "mnist"

    @property
    def task_type(self) -> TaskType:
        """このデータセットが主に属するタスク種別。"""
        if self is Dataset.MNIST:
            return TaskType.CLASSIFICATION
        return TaskType.REGRESSION

    def __str__(self) -> str:
        return self.value


class Preprocessing(enum.Enum):
    """前処理の種類（将来的な拡張も想定）。"""

    RAW = "raw"
    PCA = "pca"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ExperimentIdentifier:
    """単一の実験をユニークに識別するためのキー。"""

    pipeline: Pipeline
    dataset: Dataset
    preprocessing: Preprocessing

    def __str__(self) -> str:
        """実験IDとして使える文字列を生成 (例: mnist-fnn-b-dash-raw)。"""
        return f"{self.dataset.value}-{self.pipeline.value}-{self.preprocessing.value}"

    @classmethod
    def from_strings(
        cls,
        pipeline: str,
        dataset: str,
        preprocessing: str,
    ) -> ExperimentIdentifier:
        """文字列から ExperimentIdentifier インスタンスを生成する。"""
        return cls(
            pipeline=Pipeline(pipeline),
            dataset=Dataset(dataset),
            preprocessing=Preprocessing(preprocessing),
        )


def generate_valid_experiments() -> List[ExperimentIdentifier]:
    """実行可能な実験組み合わせをすべて生成する簡易ユーティリティ。

    現状は一例として、以下のような制約を入れている:
    - 量子モデルは MNIST では使わない（分類は classical / FNN 系が担当）
    - PCA 前処理は MNIST のときだけ有効とする
    """
    all_combinations = itertools.product(Pipeline, Dataset, Preprocessing)

    valid_experiments: List[ExperimentIdentifier] = []
    for pipeline, dataset, preprocessing in all_combinations:
        # 例1: 量子モデルは MNIST を扱わない（将来サポートするならここを緩和）
        if pipeline in {Pipeline.GATEBASED_QUANTUM, Pipeline.ANALOG_QUANTUM} and dataset is Dataset.MNIST:
            continue

        # 例2: PCA は MNIST のときのみ意味がある、と仮定
        if preprocessing is Preprocessing.PCA and dataset is not Dataset.MNIST:
            continue

        valid_experiments.append(
            ExperimentIdentifier(pipeline=pipeline, dataset=dataset, preprocessing=preprocessing)
        )

    return valid_experiments


if __name__ == "__main__":
    # 簡単な動作確認
    print("--- Individual Identifier Example ---")
    exp_id = ExperimentIdentifier(
        pipeline=Pipeline.FNN_B_DASH,
        dataset=Dataset.MNIST,
        preprocessing=Preprocessing.RAW,
    )
    print(f"Identifier object: {exp_id!r}")
    print(f"String representation: {str(exp_id)}")
    print(f"Task type: {exp_id.dataset.task_type}")

    print("\n--- Generating All Valid Experiments ---")
    for exp in generate_valid_experiments():
        print(exp)

