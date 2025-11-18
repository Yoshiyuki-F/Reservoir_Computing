"""Readout strategy protocol and result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, List, Dict, Optional

import jax.numpy as jnp


@dataclass
class ReadoutResult:
    """Container for readout training outcomes."""

    weights: jnp.ndarray
    best_lambda: float
    score_name: str
    score_val: float
    logs: List[Dict[str, float]]


class BaseReadout(Protocol):
    """Common readout interface."""

    def fit(
        self,
        X,
        Y,
        *,
        classification: bool,
        lambdas: Optional[Sequence[float]] = None,
        cv: str = "holdout",
        n_folds: int = 5,
        random_state: Optional[int] = None,
    ) -> ReadoutResult:
        ...

    def predict(self, X):
        ...

