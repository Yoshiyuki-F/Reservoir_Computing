"""src/reservoir/models/reservoir/classical/config.py"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict

from reservoir.core.identifiers import AggregationMode
from reservoir.models.config import (
    PreprocessingConfig,
    ProjectionConfig,
    ReservoirDynamicsConfig,
    AggregationConfig
)

@dataclass(frozen=True)
class ClassicalReservoirConfig:
    """
    Configuration for classical reservoir nodes.
    Hierarchically structured to match the V2.1 Pipeline Steps (2, 3, 5, 6).
    """
    preprocess: PreprocessingConfig
    projection: ProjectionConfig
    dynamics: ReservoirDynamicsConfig
    aggregation: AggregationConfig

    @property
    def n_units(self) -> int:
        """Shortcut for convenience / backward compatibility."""
        return self.projection.n_units

    @property
    def leak_rate(self) -> float:
        """Shortcut for convenience."""
        return self.dynamics.leak_rate

    def validate(self, *, context: str = "") -> None:
        """
        Basic validation hook to align with DistillationConfig expectations.
        """
        self.projection.validate(context=f"{context}projection")
        self.dynamics.validate(context=f"{context}dynamics")
        self.aggregation.validate(context=f"{context}aggregation")

    def to_dict(self) -> Dict[str, Any]:
        """
        Flattens the hierarchical config into a single dictionary.
        This maintains compatibility with Factories that expect flat parameters.
        """
        flat_dict = {}
        # Merge all sub-configs
        flat_dict.update(asdict(self.preprocess))
        flat_dict.update(asdict(self.projection))
        flat_dict.update(asdict(self.dynamics))

        # Handle AggregationMode enum serialization
        agg_data = asdict(self.aggregation)
        if isinstance(agg_data.get("mode"), AggregationMode):
            flat_dict["state_aggregation"] = agg_data["mode"].value
        else:
            flat_dict["state_aggregation"] = agg_data.get("mode")

        return flat_dict
