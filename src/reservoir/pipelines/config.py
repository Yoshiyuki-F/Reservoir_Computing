# /home/yoshi/PycharmProjects/Reservoir/src/reservoir/pipelines/config.py
from __future__ import annotations

from dataclasses import dataclass

from reservoir.core.identifiers import Dataset
from reservoir.core.interfaces import ReadoutModule
from reservoir.data import SplitDataset
from reservoir.data.config import DatasetPreset
from reservoir.training import TrainingConfig


from reservoir.layers.preprocessing import Preprocessor
from reservoir.layers.projection import Projection


@dataclass(frozen=True)
class FrontendContext:
    processed_split: SplitDataset
    preprocessor: Preprocessor | None  # Single Preprocessor instance
    preprocessed_shape: tuple[int, ...]
    projected_shape: tuple[int, ...] | None
    input_shape_for_meta: tuple[int, ...]
    input_dim_for_factory: int
    projection_layer: Projection | None = None


@dataclass(frozen=True)
class DatasetMetadata:
    dataset: Dataset
    dataset_name: str
    preset: DatasetPreset
    training: TrainingConfig
    classification: bool
    input_shape: tuple[int, ...]


from reservoir.models.generative import ClosedLoopGenerativeModel
from reservoir.core.types import ConfigDict


@dataclass(frozen=True)
class ModelStack:
    model: ClosedLoopGenerativeModel
    readout: ReadoutModule | None
    topo_meta: ConfigDict
    metric: str
    model_label: str
