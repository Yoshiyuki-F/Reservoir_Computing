"""Configuration container for classical reservoirs (hierarchical V2.1 layout)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

from reservoir.core.identifiers import AggregationMode
from reservoir.models.config import (
    PreprocessingConfig,
    ProjectionConfig,
    ReservoirDynamicsConfig,
    AggregationConfig,
)


@dataclass(frozen=True)
class ClassicalReservoirConfig:
    """Hierarchical config spanning Steps 2, 3, 5, and 6."""

    preprocess: PreprocessingConfig
    projection: ProjectionConfig
    dynamics: ReservoirDynamicsConfig
    aggregation: AggregationConfig


    # --- Validation & Serialization ---
    def validate(self, context: str = "reservoir") -> None:
        self.preprocess.validate(context=f"{context}.preprocess")
        self.projection.validate(context=f"{context}.projection")
        self.dynamics.validate(context=f"{context}.dynamics")
        self.aggregation.validate(context=f"{context}.aggregation")

    def to_dict(self) -> Dict[str, Any]:
        """
        Flatten hierarchical config for legacy consumers expecting flat dicts.
        """
        flat_dict: Dict[str, Any] = {}
        flat_dict.update(self.preprocess.to_dict())
        flat_dict.update(self.projection.to_dict())
        flat_dict.update(self.dynamics.to_dict())
        agg_mode = self.aggregation.mode
        flat_dict["state_aggregation"] = agg_mode.value if isinstance(agg_mode, AggregationMode) else agg_mode
        return flat_dict
