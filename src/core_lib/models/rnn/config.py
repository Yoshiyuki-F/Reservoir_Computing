"""Configuration model for a simple RNN."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SimpleRNNConfig(BaseModel):
    """Architecture settings for SimpleRNN."""

    input_dim: int = Field(..., gt=0)
    hidden_dim: int = Field(..., gt=0)
    output_dim: int = Field(..., gt=0)
    return_sequences: bool = False
    return_hidden: bool = False

    model_config = ConfigDict(extra="forbid")
