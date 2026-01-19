# /home/yoshi/PycharmProjects/Reservoir/src/reservoir/pipelines/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

from reservoir.core.identifiers import Dataset
from reservoir.core.interfaces import ReadoutModule
from reservoir.data import SplitDataset
from reservoir.data.config import DatasetPreset
from reservoir.training import TrainingConfig


@dataclass(frozen=True)
class FrontendContext:
    processed_split: SplitDataset
    preprocess_labels: list[str]
    preprocessors: list[Any]
    preprocessed_shape: tuple[int, ...]
    projected_shape: Optional[tuple[int, ...]]
    input_shape_for_meta: tuple[int, ...]
    input_dim_for_factory: int
    scaler: Optional[Any]
    projection_layer: Optional[Any] = None
    runtime_shapes: Dict[str, Any] = None


@dataclass(frozen=True)
class DatasetMetadata:
    dataset: Dataset
    dataset_name: str
    preset: DatasetPreset
    training: TrainingConfig
    classification: bool
    input_shape: tuple[int, ...]


@dataclass(frozen=True)
class ModelStack:
    model: Any
    readout: Optional[ReadoutModule]
    topo_meta: Dict[str, Any]
    metric: str
    model_label: str
