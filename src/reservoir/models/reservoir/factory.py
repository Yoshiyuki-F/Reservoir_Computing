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
            raise ValueError("input_shape must be provided.")
        
        # Handle 2D (Time, Feat) or 3D (Batch, Time, Feat)
        if len(input_shape) == 2:
             t_steps = int(input_shape[0])
             batch_dim = None
        elif len(input_shape) == 3:
             batch_dim = int(input_shape[0])
             t_steps = int(input_shape[1])
        else:
            raise ValueError(f"input_shape must be 2D or 3D, got {input_shape}")

        topo_meta: Dict[str, Any] = {}
        agg_mode_enum = model.aggregation

        feature_units = int(node.get_feature_dim(time_steps=t_steps))
        
        # Determine shapes with batch dimension if present
        def _with_batch(shape_wo_batch: tuple[int, ...]) -> tuple[int, ...]:
             if batch_dim is not None:
                 return (batch_dim,) + shape_wo_batch
             return shape_wo_batch

        # Correction for SEQUENCE mode: Feature shape includes time dimension
        from reservoir.core.identifiers import AggregationMode
        if agg_mode_enum == AggregationMode.SEQUENCE:
             feat_core = (t_steps, feature_units)
             out_core = (t_steps, output_dim)
        else:
             feat_core = (feature_units,)
             out_core = (output_dim,)
             
        feature_shape = _with_batch(feat_core)
        projected_shape = _with_batch((t_steps, projected_input_dim))
        output_shape = _with_batch(out_core)

        topo_meta["type"] = pipeline_config.model_type.value.upper()
        topo_meta["shapes"] = {
            "input": input_shape,
            "preprocessed": None,
            "projected": projected_shape,
            "internal": projected_shape,
            "feature": feature_shape,
            "output": output_shape,
        }
        topo_meta["details"] = {
            "preprocess": None,
            "agg_mode": agg_mode_enum.value,
            "student_layers": None,
        }
        node.topology_meta = topo_meta
        return node
