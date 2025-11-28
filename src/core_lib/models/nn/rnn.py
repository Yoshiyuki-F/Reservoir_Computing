"""/home/yoshi/PycharmProjects/Reservoir/src/core_lib/models/nn/rnn.py
RNN BaseModel wrapper using BaseFlaxModel."""

from __future__ import annotations

from typing import Any, Dict

import flax.linen as nn

from core_lib.models.nn.base import BaseFlaxModel
from core_lib.models.nn.modules import SimpleRNN as SimpleRNNModule


class RNNModel(BaseFlaxModel):
    """Wrap SimpleRNN module with BaseModel API."""

    def __init__(self, config: Dict[str, Any]):
        self.input_dim: int = int(config.get("input_dim", 1))
        self.hidden_dim: int = int(config.get("hidden_dim", 64))
        self.output_dim: int = int(config.get("output_dim", 1))
        self.return_sequences: bool = bool(config.get("return_sequences", False))
        self.return_hidden: bool = bool(config.get("return_hidden", False))
        super().__init__(config)

    def _create_model_def(self) -> nn.Module:
        return SimpleRNNModule(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            return_sequences=self.return_sequences,
            return_hidden=self.return_hidden,
        )
