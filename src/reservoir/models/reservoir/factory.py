"""src/reservoir/models/reservoir/factory.py
Unified Factory for Reservoir Domain.
STEP 5 and 6 (4 is skipped(no flatten needed))
Handles creation of Physical Nodes (Reservoir) and assembly into sequential pipelines (Steps 5-6).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import jax.numpy as jnp

from reservoir.core.identifiers import AggregationMode
from reservoir.models.config import ClassicalReservoirConfig
from reservoir.models.presets import PipelineConfig
from reservoir.models.reservoir.base import Reservoir
from reservoir.models.reservoir.classical import ClassicalReservoir


class ReservoirFactory:
    """Factory for creating Reservoir Nodes (Steps 5-6)."""

    @staticmethod
    def create_pipeline(
        pipeline_config: PipelineConfig,
        input_dim: int,
        output_dim: int,
        input_shape: Optional[tuple[int, ...]],
    ) -> Reservoir:
        """
        Assemble reservoir node with embedded aggregation (Steps 5-6).
        Assumes inputs are already projected to the reservoir dimensionality (input_dim).
        """
        if not isinstance(pipeline_config, PipelineConfig):
            raise TypeError(f"ReservoirFactory expects PipelineConfig, got {type(pipeline_config)}.")
        model = pipeline_config.model
        if not isinstance(model, ClassicalReservoirConfig):
            raise TypeError(f"ReservoirFactory requires ClassicalReservoirConfig, got {type(model)}.")

        projected_input_dim = int(input_dim)
        node = ReservoirFactory.create_node(model, projected_input_dim)

        # Topology metadata
        topo_meta: Dict[str, Any] = {}
        in_shape = input_shape
        if input_shape is None:
            t_steps = 1
        elif len(input_shape) == 1:
            t_steps = int(input_shape[0])
        elif len(input_shape) == 2:
            t_steps = int(input_shape[0])
        else:
            t_steps = int(input_shape[-2])
        agg_mode_enum = model.aggregation

        feature_units = int(node.get_feature_dim(time_steps=t_steps))

        topo_meta["type"] = pipeline_config.model_type.value.upper()
        topo_meta["shapes"] = {
            "input": in_shape,
            "preprocessed": None,
            "projected": (t_steps, projected_input_dim) if in_shape else None,
            "internal": (t_steps, projected_input_dim),
            "feature": (int(feature_units),),
            "output": (int(output_dim),),
        }
        topo_meta["details"] = {
            "preprocess": None,
            "agg_mode": agg_mode_enum.value if isinstance(agg_mode_enum, AggregationMode) else str(agg_mode_enum),
            "student_layers": None,
        }
        node.topology_meta = topo_meta
        return node



    @staticmethod  # for distillation use
    def create_node(config: ClassicalReservoirConfig, input_dim: int) -> Reservoir:
        """
        Low-level method to create just the Reservoir Node.
        Used internally and by DistillationFactory (to create Teacher).
        """
        return ClassicalReservoir(
            n_units=input_dim,
            spectral_radius=config.spectral_radius,
            leak_rate=config.leak_rate,
            rc_connectivity=config.rc_connectivity,
            seed=config.seed,
            aggregation_mode=config.aggregation,
        )
