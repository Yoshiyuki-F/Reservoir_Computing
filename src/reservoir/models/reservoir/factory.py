"""src/reservoir/models/reservoir/factory.py
Unified Factory for Reservoir Domain.
STEP 5 and 6 (4 is skipped(no flatten needed))
Handles creation of Physical Nodes (Reservoir) and assembly into sequential pipelines (Steps 5-6).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from reservoir.models.config import ClassicalReservoirConfig, QuantumReservoirConfig
from reservoir.models.presets import PipelineConfig
from reservoir.models.reservoir.classical import ClassicalReservoir
from reservoir.models.reservoir.quantum import QuantumReservoir
from reservoir.models.reservoir.base import Reservoir


class ReservoirFactory:
    """Factory for creating Reservoir Nodes (Steps 5-6)."""

    @staticmethod
    def create_model(
        pipeline_config: PipelineConfig,
        projected_input_dim: int,
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
        
        if isinstance(model, ClassicalReservoirConfig):
            node = ClassicalReservoir(
                n_units=projected_input_dim,
                spectral_radius=model.spectral_radius,
                leak_rate=model.leak_rate,
                rc_connectivity=model.rc_connectivity,
                seed=model.seed,
                aggregation_mode=model.aggregation,
            )
        elif isinstance(model, QuantumReservoirConfig):
            # n_qubits implies the size of the projected input vector (Step 3)
            # which is `projected_input_dim`
            node = QuantumReservoir(
                n_qubits=projected_input_dim,
                n_layers=model.n_layers,
                seed=model.seed,
                aggregation_mode=model.aggregation,
                leak_rate=model.leak_rate,
                feedback_scale=model.feedback_scale,
                measurement_basis=model.measurement_basis,
                encoding_strategy=model.encoding_strategy,
                noise_type=model.noise_type,
                noise_prob=model.noise_prob,
                readout_error=model.readout_error,
                n_trajectories=model.n_trajectories,
                use_remat=model.use_remat,
                use_reuploading=model.use_reuploading,
                precision=model.precision,
            )
        else:
            raise TypeError(f"ReservoirFactory requires Classical or Quantum config, got {type(model)}.")

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

        # Determine shapes with batch dimension if present
        def _with_batch(shape_wo_batch: tuple[int, ...]) -> tuple[int, ...]:
             if batch_dim is not None:
                 return (batch_dim,) + shape_wo_batch
             return shape_wo_batch

        # Use Aggregator execution logic to determine feature shape
        from reservoir.layers.aggregation import StateAggregator
        
        # internal_shape is the output of the Reservoir (Step 5), which is input to Aggregation (Step 6)
        # Use node.n_units because QuantumReservoir (and others) might expand dimension (e.g., 4 -> 10)
        internal_dim = node.n_units
        internal_shape = _with_batch((t_steps, internal_dim))
        
        aggregator = StateAggregator(agg_mode_enum)
        feature_shape = aggregator.get_output_shape(internal_shape)
        
        # Output shape matches feature shape but with last dimension replaced by output_dim
        # (Assuming Readout preserves sample structure and maps features -> output_dim)
        output_shape = feature_shape[:-1] + (output_dim,)
             
        # projected_shape is the input to the Reservoir (Step 3/4)
        projected_shape = _with_batch((t_steps, projected_input_dim))

        topo_meta["type"] = pipeline_config.model_type.value.upper()
        topo_meta["shapes"] = {
            "input": input_shape,
            "preprocessed": None,
            "projected": projected_shape,
            "adapter": None,  # No adapter for reservoir (direct sequence processing)
            "internal": internal_shape,
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
