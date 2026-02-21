"""
src/reservoir/readout/poly_ridge.py
Polynomial feature expansion readout – inherits RidgeCV without modifying it.

Both modes are implemented with **pure JAX operations** so they work
inside jax.lax.scan (closed-loop generation).

Two modes:
  - "square_only": appends x_i^2 (and optionally x_i^3, …) to the original vector.
    Keeps dimensionality manageable (N → N * degree).
  - "full": all cross-terms x_i * x_j (i <= j) via jnp upper-triangle indexing.
    Produces N + N*(N+1)/2 features for degree=2.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING
import jax.numpy as jnp

from reservoir.readout.ridge import RidgeCV

if TYPE_CHECKING:
    from reservoir.core.types import JaxF64, ConfigDict, NpF64
    from collections.abc import Callable


class PolyRidgeReadout(RidgeCV):
    """Ridge readout with polynomial feature expansion.

    Overrides fit / predict / fit_with_validation to expand features
    *before* delegating to the parent RidgeCV logic.
    All expansion is pure JAX – safe inside jax.lax.scan.
    """

    def __init__(
        self,
        lambda_candidates: tuple[float, ...],
        use_intercept: bool = True,
        degree: int = 2,
        mode: Literal["full", "square_only"] = "square_only",
    ) -> None:
        super().__init__(lambda_candidates=lambda_candidates, use_intercept=use_intercept)
        self.degree = degree
        self.mode = mode

    # ------------------------------------------------------------------
    # Feature expansion (pure JAX)
    # ------------------------------------------------------------------
    def _expand_features(self, X: JaxF64) -> JaxF64:
        """Expand input features according to the configured mode."""
        if self.mode == "square_only":
            return self._expand_square_only(X)
        elif self.mode == "full":
            return self._expand_full(X)
        else:
            raise ValueError(f"Unknown PolyRidgeReadout mode: {self.mode!r}")

    def _expand_square_only(self, X: JaxF64) -> JaxF64:
        """Append x_i^k for k=2..degree to the original feature vector.

        For degree=2:  [x1, ..., xN, x1^2, ..., xN^2]
        """
        parts = [X]
        for k in range(2, self.degree + 1):
            parts.append(X ** k)
        return jnp.concatenate(parts, axis=-1)

    def _expand_full(self, X: JaxF64) -> JaxF64:
        """Pure-JAX full polynomial expansion (degree=2).

        Produces: [original features] + [x_i * x_j for i <= j]
        For n features → n + n*(n+1)/2 output features.
        """
        n_features = X.shape[-1]

        # Upper-triangle indices (including diagonal) → x_i * x_j for i <= j
        idx_i, idx_j = jnp.triu_indices(n_features)
        cross_terms = X[..., idx_i] * X[..., idx_j]  # works for any batch dims

        return jnp.concatenate([X, cross_terms], axis=-1)

    # ------------------------------------------------------------------
    # Overridden ReadoutModule interface
    # ------------------------------------------------------------------
    def fit(self, states: JaxF64, targets: JaxF64) -> PolyRidgeReadout:
        """Expand features, then delegate to RidgeCV.fit."""
        X_expanded = self._expand_features(states)
        super().fit(X_expanded, targets)
        return self

    def predict(self, states: JaxF64) -> JaxF64:
        """Expand features, then delegate to RidgeCV.predict."""
        X_expanded = self._expand_features(states)
        return super().predict(X_expanded)

    def fit_with_validation(
        self,
        train_Z: JaxF64,
        train_y: JaxF64,
        val_Z: JaxF64,
        val_y: JaxF64,
        scoring_fn: Callable,
        maximize_score: bool = True,
    ) -> tuple[float, float, dict[float, float], dict[float, float], dict[float, NpF64]]:
        """Expand features for both train and val, then delegate to RidgeCV."""
        from reservoir.utils.reporting import print_feature_stats

        train_expanded = self._expand_features(train_Z)
        val_expanded = self._expand_features(val_Z)

        print(
            f"    [PolyRidge] mode={self.mode}, degree={self.degree} | "
            f"input_dim={train_Z.shape[-1]} → expanded_dim={train_expanded.shape[-1]}"
        )
        print_feature_stats(train_expanded, "7:poly_expanded_train")
        print_feature_stats(val_expanded, "7:poly_expanded_val")

        return super().fit_with_validation(
            train_Z=train_expanded,
            train_y=train_y,
            val_Z=val_expanded,
            val_y=val_y,
            scoring_fn=scoring_fn,
            maximize_score=maximize_score,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> ConfigDict:
        data = super().to_dict()
        res: ConfigDict = dict(data)
        res["degree"] = int(self.degree)
        res["mode"] = str(self.mode)
        return res
