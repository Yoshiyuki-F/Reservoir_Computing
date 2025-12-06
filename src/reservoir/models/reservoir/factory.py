"""src/reservoir/models/reservoir/factory.py
Unified Factory for Reservoir Domain.
STEP 5 and 6 (4 is skipped(no flatten needed))
Handles creation of Physical Nodes (Reservoir) and assembly into sequential pipelines (Steps 5-6).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, List

from reservoir.core.identifiers import Pipeline, AggregationMode
from reservoir.layers.aggregation import StateAggregator
from reservoir.models.reservoir.base import Reservoir
from reservoir.models.reservoir.classical import ClassicalReservoir
from reservoir.models.reservoir.classical.config import ClassicalReservoirConfig
from reservoir.models.sequential import SequentialModel


class ReservoirFactory:
    """Factory for creating Reservoir Nodes and sequential Reservoir models."""

    @staticmethod
    def create_pipeline(
        reservoir_params: Dict[str, Any] | ClassicalReservoirConfig,
        *,
        input_dim: int,
        output_dim: int,
        input_shape: Optional[tuple[int, ...]] = None,
        pipeline: Pipeline = Pipeline.CLASSICAL_RESERVOIR,
    ) -> SequentialModel:
        """
        Assemble reservoir node and aggregation into SequentialModel (Steps 5-6).
        Assumes inputs are already projected to reservoir_config.n_units dimensionality.
        """
        reservoir_config = (
            reservoir_params
            if isinstance(reservoir_params, ClassicalReservoirConfig)
            else ClassicalReservoirConfig(**reservoir_params)
        )
        reservoir_config.validate(context=pipeline.value)

        projected_input_dim = int(input_dim)
        node = ReservoirFactory.create_node(
            reservoir_config,
            projected_input_dim,
            use_input_projection=False,
        )

        layers: List[Any] = []
        layers.append(node)
        aggregator = StateAggregator(mode=reservoir_config.state_aggregation)
        layers.append(aggregator)

        seq = SequentialModel(layers)
        seq.reservoir = node
        seq.aggregator = aggregator
        seq.effective_input_dim = projected_input_dim

        # Topology metadata
        topo_meta: Dict[str, Any] = {}
        in_shape = input_shape or (1, projected_input_dim)
        t_steps = in_shape[0] if len(in_shape) > 1 else 1
        agg_mode_enum = reservoir_config.state_aggregation
        feature_units = reservoir_config.n_units
        if agg_mode_enum in {AggregationMode.LAST_MEAN, AggregationMode.MTS}:
            feature_units = reservoir_config.n_units * 2
        elif agg_mode_enum == AggregationMode.CONCAT:
            feature_units = reservoir_config.n_units * t_steps

        topo_meta["type"] = pipeline.value.upper()
        topo_meta["shapes"] = {
            "input": None,
            "preprocessed": None,
            "projected": None,
            "internal": (t_steps, reservoir_config.n_units),
            "feature": (int(feature_units),),
            "output": (int(output_dim),),
        }
        topo_meta["details"] = {
            "preprocess": None,
            "agg_mode": agg_mode_enum.value if isinstance(agg_mode_enum, AggregationMode) else str(agg_mode_enum),
            "student_layers": None,
        }
        seq.topology_meta = topo_meta
        return seq

    @staticmethod  # for distillation use
    def create_node(
        config: ClassicalReservoirConfig,
        input_dim: int,
        *,
        use_input_projection: bool = False,
    ) -> Reservoir:
        """
        Low-level method to create just the Reservoir Node.
        Used internally and by DistillationFactory (to create Teacher).
        """
        config.validate(context="reservoir")

        seed_val = 0 if config.seed is None else int(config.seed)

        return ClassicalReservoir(
            n_inputs=int(input_dim),
            n_units=int(config.n_units),
            spectral_radius=float(config.spectral_radius),
            leak_rate=float(config.leak_rate),
            rc_connectivity=float(config.rc_connectivity),
            noise_rc=float(config.noise_rc),
            seed=seed_val,
        )
