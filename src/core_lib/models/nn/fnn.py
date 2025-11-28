"""/home/yoshi/PycharmProjects/Reservoir/src/core_lib/models/nn/fnn.py
FNN BaseModel wrapper using BaseFlaxModel."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import flax.linen as nn

from core_lib.models.nn.base import BaseFlaxModel
from core_lib.models.nn.modules import FNN as FNNModule


class FNNModel(BaseFlaxModel):
    """Wrap FNNModule with BaseModel API."""

    def __init__(self, config: Dict[str, Any]):
        self.layer_dims: Sequence[int] = config.get("layer_dims", [])
        super().__init__(config)

    def _create_model_def(self) -> nn.Module:
        return FNNModule(layer_dims=self.layer_dims, return_hidden=False)
