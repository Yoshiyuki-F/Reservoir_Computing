"""Configuration models for FNN pipeline (b)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from pydantic import BaseModel, ConfigDict, Field


class FNNTrainingConfig(BaseModel):
    """Settings for the FNN pretraining phase."""

    learning_rate: float = Field(..., gt=0.0)
    batch_size: int = Field(..., gt=0)
    num_epochs: int = Field(..., gt=0)
    weights_path: str

    model_config = ConfigDict(extra="forbid")


class FNNModelConfig(BaseModel):
    """Architecture configuration for the FNN model."""

    layer_dims: List[int]

    model_config = ConfigDict(extra="forbid")

    @property
    def input_dim(self) -> int:
        if not self.layer_dims:
            raise ValueError("layer_dims must contain at least input and output dimensions")
        return self.layer_dims[0]

    @property
    def hidden_dims(self) -> List[int]:
        if len(self.layer_dims) < 2:
            raise ValueError("layer_dims must contain at least one output dimension")
        return self.layer_dims[1:]


class FNNPipelineConfig(BaseModel):
    """Complete configuration for FNN pipeline (b)."""

    model: FNNModelConfig
    training: FNNTrainingConfig
    ridge_lambdas: Union[List[float], Dict[str, Any]] = Field(
        ...,
        description=(
            "Ridge regularization search space. "
            "Supports [start, stop, num] (log10) or explicit list."
        ),
    )
    use_preprocessing: bool = Field(
        ...,
        description="Whether to apply FeatureScaler + DesignMatrixBuilder before ridge readout",
    )

    model_config = ConfigDict(extra="forbid")

    @property
    def weights_path(self) -> Path:
        return Path(self.training.weights_path)
