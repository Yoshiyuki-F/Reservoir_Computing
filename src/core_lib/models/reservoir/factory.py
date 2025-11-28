"""Factory for creating reservoir models."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple, Union

from .base import ReservoirComputerFactory


class ReservoirFactory:
    """Centralized creator for reservoir computing models (classical/quantum)."""

    @staticmethod
    def create(
        config: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
        input_shape: Tuple[int, ...],
        *,
        reservoir_type: str | None = None,
        backend: str | None = None,
    ) -> Any:
        """Instantiate a reservoir model based on config and input shape.

        Args:
            config: Reservoir configuration dict or sequence of dicts.
            input_shape: Shape of the input data (e.g., (time, features) or (features,)).
            reservoir_type: Optional explicit reservoir type. Falls back to config.
            backend: Optional backend for ReservoirComputerFactory.
        """
        # Normalize to list form expected by ReservoirComputerFactory
        cfg = config
        if isinstance(config, dict):
            cfg = [config]

        n_inputs = input_shape[-1]

        # ensure n_inputs/n_outputs present in configs
        for item in cfg:  # type: ignore[union-attr]
            params = item.setdefault("params", {}) if isinstance(item, dict) else {}
            if isinstance(params, dict):
                params.setdefault("n_inputs", n_inputs)
                # Default to log-spaced grid [-7, 5, 17] -> 1e-7 ... 1e5 with 17 points
                params.setdefault("ridge_lambdas", [-7, 5, 17])
                params.setdefault("state_aggregation", params.get("state_aggregation", "mean"))

        r_type = reservoir_type
        if r_type is None:
            # look for name or reservoir_type hints
            for item in cfg:  # type: ignore[union-attr]
                if isinstance(item, dict):
                    if "name" in item:
                        r_type = item["name"]
                        break
                    if "reservoir_type" in item:
                        r_type = item["reservoir_type"]
                        break
        if r_type is None:
            r_type = "classical"

        return ReservoirComputerFactory.create_reservoir(
            reservoir_type=r_type,
            config=cfg,
            backend=backend or "cpu",
        )
