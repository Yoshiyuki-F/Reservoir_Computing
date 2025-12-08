"""src/reservoir/models/reservoir/factory.py
Unified Factory for Reservoir Domain.
STEP 5 and 6 (4 is skipped(no flatten needed))
Handles creation of Physical Nodes (Reservoir) and assembly into sequential pipelines (Steps 5-6).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from reservoir.models.config import ClassicalReservoirConfig
from reservoir.models.presets import PipelineConfig
from reservoir.models.reservoir.classical import ClassicalReservoir


class ReservoirFactory:
    """Factory for creating Reservoir Nodes (Steps 5-6)."""

    @staticmethod
    def create_pipeline(
        pipeline_config: PipelineConfig,
        projected_input_dim: int,
        output_dim: int,
        input_shape: Optional[tuple[int, ...]],
    ) -> ClassicalReservoir:
        """
        Assemble reservoir node with embedded aggregation (Steps 5-6).
        Assumes inputs are already projected to the reservoir dimensionality (input_dim).
        """
        if not isinstance(pipeline_config, PipelineConfig):
            raise TypeError(f"ReservoirFactory expects PipelineConfig, got {type(pipeline_config)}.")
        model = pipeline_config.model
        if not isinstance(model, ClassicalReservoirConfig):
            raise TypeError(f"ReservoirFactory requires ClassicalReservoirConfig, got {type(model)}.")

        node = ClassicalReservoir(
            n_units=projected_input_dim,
            spectral_radius=model.spectral_radius,
            leak_rate=model.leak_rate,
            rc_connectivity=model.rc_connectivity,
            seed=model.seed,
            aggregation_mode=model.aggregation,
        )

        if input_shape is None:
            raise ValueError("input_shape must be provided to ReservoirFactory.create_pipeline (time, features).")
        if len(input_shape) != 2:
            raise ValueError(f"input_shape must be (time, features), got {input_shape}")
        t_steps = int(input_shape[0])
        topo_meta: Dict[str, Any] = {}
        agg_mode_enum = model.aggregation

        feature_units = int(node.get_feature_dim(time_steps=t_steps))

        topo_meta["type"] = pipeline_config.model_type.value.upper()
        topo_meta["shapes"] = {
            "input": input_shape,
            "preprocessed": None,
            "projected": (t_steps, projected_input_dim),
            "internal": (t_steps, projected_input_dim),
            "feature": (feature_units,),
            "output": (output_dim,),
        }
        topo_meta["details"] = {
            "preprocess": None,
            "agg_mode": agg_mode_enum.value,
            "student_layers": None,
        }
        node.topology_meta = topo_meta
        return node
