"""src/reservoir/models/reservoir/factory.py
Unified Factory for Reservoir Domain.
Handles creation of Physical Nodes (Reservoir) and Logical Models (ReservoirModel).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from reservoir.components.projection import InputProjector
from reservoir.components import FeatureScaler, DesignMatrix, TransformerSequence
from reservoir.models.reservoir.base import Reservoir
from reservoir.models.reservoir.classical import ClassicalReservoir
from reservoir.models.reservoir.classical.config import ClassicalReservoirConfig
from reservoir.models.reservoir.model import ReservoirModel


class ReservoirFactory:
    """Factory for creating Reservoir Nodes and full ReservoirModels."""

    @staticmethod
    def create_model(config: Dict[str, Any]) -> ReservoirModel:
        """
        Creates a full ReservoirModel, including Preprocessing and Design Matrix.
        """
        res_cfg_dict = config.get("reservoir") or config.get("reservoir_params")
        if not res_cfg_dict:
            res_cfg_dict = {k: v for k, v in config.items() if hasattr(ClassicalReservoirConfig, k)}
        reservoir_config = ClassicalReservoirConfig(**res_cfg_dict)

        raw_input_dim = config.get("input_dim")
        if raw_input_dim is None:
            raise ValueError("ReservoirModel requires 'input_dim'.")
        raw_input_dim = int(raw_input_dim)

        preprocess_steps = []
        effective_input_dim = raw_input_dim

        if config.get("use_preprocessing", True):
            preprocess_steps.append(FeatureScaler())

        if reservoir_config.use_design_matrix:
            degree = reservoir_config.poly_degree
            include_bias = bool(config.get("poly_bias", True))
            factor = degree if degree > 0 else 1
            effective_input_dim = effective_input_dim * factor + (1 if include_bias else 0)
            preprocess_steps.append(DesignMatrix(degree=degree, include_bias=include_bias))

        preprocessor = TransformerSequence(preprocess_steps) if preprocess_steps else None

        node = ReservoirFactory.create_node(reservoir_config, int(effective_input_dim))

        return ReservoirModel(
            reservoir=node,
            preprocess=preprocessor,
            readout_mode=reservoir_config.state_aggregation or "mean",
        )

    @staticmethod  # for distillation use
    def create_node(
        config: ClassicalReservoirConfig,
        input_dim: int,
        projector: Optional[InputProjector] = None,
    ) -> Reservoir:
        """
        Low-level method to create just the Reservoir Node.
        Used internally and by DistillationFactory (to create Teacher).
        """
        config.validate(context="reservoir")

        return ClassicalReservoir(
            n_inputs=int(input_dim),
            n_units=int(config.n_units),
            input_scale=float(config.input_scale),
            spectral_radius=float(config.spectral_radius),
            leak_rate=float(config.leak_rate),
            input_connectivity=float(config.input_connectivity),
            rc_connectivity=float(config.rc_connectivity),
            noise_rc=float(config.noise_rc),
            bias_scale=float(config.bias_scale),
            seed=int(config.seed),
            projector=projector,
        )


# Backward-compatible alias expected by older pipeline/test code
ReservoirFactory.create = staticmethod(ReservoirFactory.create_model)
