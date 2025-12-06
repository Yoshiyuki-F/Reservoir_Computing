"""src/reservoir/models/reservoir/factory.py
Unified Factory for Reservoir Domain.
Handles creation of Physical Nodes (Reservoir) and assembly into sequential pipelines.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, List, Iterable

from reservoir.core.identifiers import Pipeline, AggregationMode
from reservoir.layers.projection import InputProjection
from reservoir.layers.preprocessing import FeatureScaler, DesignMatrix
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
        use_preprocessing: bool,
        input_shape: Optional[tuple[int, ...]] = None,
        pipeline: Pipeline = Pipeline.CLASSICAL_RESERVOIR,
    ) -> SequentialModel:
        """
        Assemble preprocessing, projection, reservoir node, and aggregation into SequentialModel.
        Attaches topology_meta describing the 6-step tensor flow.
        """
        reservoir_config = (
            reservoir_params
            if isinstance(reservoir_params, ClassicalReservoirConfig)
            else ClassicalReservoirConfig(**reservoir_params)
        )
        reservoir_config.validate(context=pipeline.value)

        raw_input_dim = int(input_dim)
        preprocess_layers: List[Any] = []
        effective_input_dim = raw_input_dim

        preprocess_labels: List[str] = []
        if use_preprocessing:
            preprocess_layers.append(FeatureScaler())
            preprocess_labels.append("scaler")

        if reservoir_config.use_design_matrix:
            degree = reservoir_config.poly_degree
            include_bias = True
            factor = degree if degree > 0 else 1
            effective_input_dim = effective_input_dim * factor + (1 if include_bias else 0)
            preprocess_layers.append(DesignMatrix(degree=degree, include_bias=include_bias))
            preprocess_labels.append(f"poly{degree}")

        projection = InputProjection(
            input_dim=effective_input_dim,
            output_dim=reservoir_config.n_units,
            input_scale=reservoir_config.input_scale,
            input_connectivity=reservoir_config.input_connectivity,
            bias_scale=reservoir_config.bias_scale,
            seed=reservoir_config.seed or 0,
        )

        node = ReservoirFactory.create_node(
            reservoir_config,
            int(reservoir_config.n_units),
            use_input_projection=False,
        )

        layers: List[Any] = []
        layers.extend(preprocess_layers)
        layers.append(projection)
        layers.append(node)
        aggregator = StateAggregator(mode=reservoir_config.state_aggregation)
        layers.append(aggregator)

        seq = SequentialModel(layers)
        seq.preprocess = preprocess_layers if preprocess_layers else None
        seq.reservoir = node
        seq.aggregator = aggregator
        seq.effective_input_dim = effective_input_dim

        # Topology metadata
        topo_meta: Dict[str, Any] = {}
        in_shape = input_shape or (1, raw_input_dim)
        t_steps = in_shape[0] if len(in_shape) > 1 else 1
        agg_mode_enum = reservoir_config.state_aggregation
        feature_units = reservoir_config.n_units
        if agg_mode_enum in {AggregationMode.LAST_MEAN, AggregationMode.MTS}:
            feature_units = reservoir_config.n_units * 2
        elif agg_mode_enum == AggregationMode.CONCAT:
            feature_units = reservoir_config.n_units * t_steps

        topo_meta["type"] = pipeline.value.upper()
        topo_meta["shapes"] = {
            "input": in_shape,
            "preprocessed": (t_steps, effective_input_dim),
            "projected": (t_steps, reservoir_config.n_units),
            "internal": (t_steps, reservoir_config.n_units),
            "feature": (int(feature_units),),
            "output": (int(output_dim),),
        }
        topo_meta["details"] = {
            "preprocess": "-".join(preprocess_labels) if preprocess_labels else None,
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
