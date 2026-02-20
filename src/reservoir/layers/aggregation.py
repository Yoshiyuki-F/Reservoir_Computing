"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/layers/aggregation.py
Step 6 aggregation layers.
State aggregation components compatible with Transformer protocol.
"""

from __future__ import annotations

from typing import Dict, Optional

from beartype import beartype
import jax.numpy as jnp
from reservoir.core.types import JaxF64, ConfigDict
from reservoir.core.interfaces import Transformer
from reservoir.core.identifiers import AggregationMode

@beartype
class StateAggregator(Transformer):
    """Stateless transformer that reduces the time axis using a configured mode."""

    def __init__(self, mode: AggregationMode) -> None:
        self.mode = self._resolve_mode(mode)

    @staticmethod
    def _resolve_mode(mode: AggregationMode) -> AggregationMode:
        if isinstance(mode, AggregationMode):
            return mode
        if isinstance(mode, str):
            try:
                return AggregationMode(mode)
            except Exception as exc:
                raise ValueError(f"Invalid aggregation mode '{mode}'") from exc
        raise TypeError(f"Aggregation mode must be AggregationMode or str, got {type(mode)}.")

    @staticmethod
    def aggregate(states: JaxF64, mode: AggregationMode) -> JaxF64:
        """Static aggregator for reuse in functional contexts."""
        agg_mode = StateAggregator._resolve_mode(mode)
        arr = states
        if agg_mode is AggregationMode.SEQUENCE:
            # Flatten to 2D (Batch * Time, Features) to match readout expectation and metadata
            if arr.ndim == 3:
                return arr.reshape(-1, arr.shape[-1])
            return arr
        if arr.ndim == 3:
            if agg_mode is AggregationMode.LAST:
                return arr[:, -1, :]
            if agg_mode is AggregationMode.MEAN:
                return jnp.mean(arr, axis=1)
            if agg_mode in {AggregationMode.LAST_MEAN, AggregationMode.MTS}:
                last = arr[:, -1, :]
                mean = jnp.mean(arr, axis=1)
                return jnp.concatenate([last, mean], axis=1)
            if agg_mode is AggregationMode.CONCAT:
                return arr.reshape(arr.shape[0], -1)
        elif arr.ndim == 2:
            if agg_mode is AggregationMode.LAST:
                return arr[-1]
            if agg_mode is AggregationMode.MEAN:
                return jnp.mean(arr, axis=0)
            if agg_mode in {AggregationMode.LAST_MEAN, AggregationMode.MTS}:
                last = arr[-1]
                mean = jnp.mean(arr, axis=0)
                return jnp.concatenate([last, mean], axis=0)
            if agg_mode is AggregationMode.CONCAT:
                return arr.reshape(-1)
        raise ValueError(f"Unsupported shape {arr.shape} or aggregation mode: {agg_mode}")

    def fit(self, features: JaxF64, y: Optional[JaxF64] = None) -> "StateAggregator":
        return self

    def transform(self, features: JaxF64, log_label: Optional[str] = None) -> JaxF64:
        result = StateAggregator.aggregate(features, self.mode)
        
        # Assert output is 2D (Samples, Features) - required by readout layer
        assert result.ndim == 2, f"Aggregation output must be 2D, got {result.shape}"
        
        if log_label is not None:
            from reservoir.utils.reporting import print_feature_stats
            print_feature_stats(result, log_label)
        return result

    def fit_transform(self, features: JaxF64) -> JaxF64:
        return self.transform(features)

    def __call__(self, features: JaxF64, log_label: Optional[str] = None) -> JaxF64:
        return self.transform(features, log_label=log_label)

    def get_output_dim(self, n_units: int, n_steps: int) -> int:
        """Compute aggregated feature dimension without materializing data."""
        mode = self.mode
        steps = int(n_steps)
        units = int(n_units)
        if steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if units <= 0:
            raise ValueError(f"n_units must be positive, got {n_units}")

        if mode in {AggregationMode.LAST, AggregationMode.MEAN}:
            return units
        if mode in {AggregationMode.LAST_MEAN, AggregationMode.MTS}:
            return units * 2
        if mode is AggregationMode.CONCAT:
            return units * steps
        if mode is AggregationMode.SEQUENCE:
            return units
        raise ValueError(f"Unknown aggregation mode: {mode}")

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Compute output shape based on input shape.
        Handles (Batch, Time, Units) -> (Batch, OutputDim) or flattened (Batch*Time, Units) for Sequence.
        scan
        """
        mode = self.mode
        
        if len(input_shape) == 3:
            # (Batch, Time, Units)
            batch, steps, units = input_shape
            
            if mode is AggregationMode.SEQUENCE:
                # Flatten time dimension for sequence mode: (Batch * Time, Units)
                return batch * steps, units
            
            if mode is AggregationMode.CONCAT:
                 # (Batch, Time * Units)
                 return batch, steps * units

            # For LAST, MEAN, etc. -> (Batch, Units) (or Units*2 for bidirectional/MTS)
            out_dim = self.get_output_dim(units, steps)
            return batch, out_dim

        elif len(input_shape) == 2:
            # (Time, Units) -> Single sample context
            steps, units = input_shape
            
            if mode is AggregationMode.SEQUENCE:
                # (Time, Units) - effectively already flat
                return steps, units
            
            if mode is AggregationMode.CONCAT:
                return (steps * units,)
                
            # For LAST, MEAN -> (OutputDim,)
            out_dim = self.get_output_dim(units, steps)
            return (out_dim,)
            
        raise ValueError(f"Unsupported input shape: {input_shape}")

    def to_dict(self) -> ConfigDict:
        return {"mode": self.mode.value if isinstance(self.mode, AggregationMode) else str(self.mode)}

    @classmethod
    def from_dict(cls, data: ConfigDict) -> "StateAggregator":
        return cls(mode=AggregationMode(str(data.get("mode", AggregationMode.LAST.value))))


__all__ = ["StateAggregator"]
