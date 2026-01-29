"""
src/reservoir/components/readout/ridge.py
Ridge regression implementation compliant with ReadoutModule protocol.
"""
from __future__ import annotations

from typing import Dict, Any, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg

from reservoir.core.interfaces import ReadoutModule

# Lazy x64 enablement - only when Ridge methods are called
_X64_ENABLED = False

def _ensure_x64():
    """Enable x64 precision on first Ridge computation."""
    global _X64_ENABLED
    if not _X64_ENABLED:
        jax.config.update("jax_enable_x64", True)
        _X64_ENABLED = True


class RidgeRegression(ReadoutModule):
    """Ridge regression readout solved with JAX linear algebra."""

    def __init__(
        self,
        ridge_lambda: float,
        use_intercept: bool,
        lambda_candidates: Optional[tuple] = None,
    ) -> None:
        lambda_val = float(ridge_lambda)
        self.ridge_lambda = lambda_val
        self.use_intercept = bool(use_intercept)
        self.lambda_candidates = lambda_candidates
        self.coef_: Optional[jnp.ndarray] = None
        self.intercept_: Optional[jnp.ndarray] = None
        self.input_dim_: Optional[int] = None

    def _prepare_xy(self, states: jnp.ndarray, targets: jnp.ndarray, *, update_dim: bool) \
            -> tuple[jnp.ndarray, jnp.ndarray, bool]:
        X = jnp.asarray(states)
        if X.ndim != 2:
            raise ValueError(f"States must be 2D, got {X.shape}")
        y_arr = jnp.asarray(targets)
        y_is_1d = y_arr.ndim == 1
        if y_is_1d:
            y_arr = y_arr[:, None]
        if y_arr.shape[0] != X.shape[0]:
            raise ValueError(f"Mismatched samples: states have {X.shape[0]}, targets have {y_arr.shape[0]}.")

        n_samples, n_features = X.shape
        if update_dim:
            self.input_dim_ = n_features
        elif self.input_dim_ is not None and n_features != self.input_dim_:
            raise ValueError(f"Expected states with {self.input_dim_} features, got {n_features}.")

        if self.use_intercept:
            ones = jnp.ones((n_samples, 1))
            X = jnp.concatenate([ones, X], axis=1)
        return X, y_arr, y_is_1d

    def fit(self, states: jnp.ndarray, targets: jnp.ndarray) -> "RidgeRegression":
        """Fit a single ridge model without validation search."""
        _ensure_x64()  # Enable float64 for numerical stability
        X, y_arr, y_is_1d = self._prepare_xy(states, targets, update_dim=True)
        n_features = X.shape[1]
        eye = jnp.eye(n_features)
        XtX = X.T @ X
        Xty = X.T @ y_arr
        lam_val = float(self.ridge_lambda)
        solve_mat = XtX + lam_val * eye
        if lam_val < 1e-7:
            w = jax.scipy.linalg.solve(solve_mat, Xty)  # safer general solver for near-singular cases
        else:
            w = jax.scipy.linalg.solve(solve_mat, Xty, assume_a="pos")
        if not jnp.all(jnp.isfinite(w)):
            w = jnp.zeros_like(w)
        if self.use_intercept:
            intercept = jnp.asarray(w[0])
            coef = jnp.asarray(w[1:])
        else:
            intercept = jnp.zeros(w.shape[1])
            coef = jnp.asarray(w)
        if y_is_1d:
            coef = coef.ravel()
            intercept = intercept.ravel()
        self.coef_ = coef
        self.intercept_ = intercept
        return self

    def predict(self, states: jnp.ndarray) -> jnp.ndarray:
        if self.coef_ is None:
            raise RuntimeError("RidgeRegression is not fitted yet.")
        X = jnp.asarray(states)
        return X @ self.coef_ + self.intercept_ if self.use_intercept else X @ self.coef_

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"ridge_lambda": self.ridge_lambda, "use_intercept": self.use_intercept}
        if self.coef_ is not None:
            data["coef"] = jnp.asarray(self.coef_).tolist()
        if self.intercept_ is not None:
            data["intercept"] = jnp.asarray(self.intercept_).tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RidgeRegression":
        if "ridge_lambda" not in data:
            raise ValueError("Serialized RidgeRegression is missing required 'ridge_lambda'.")
        model = cls(ridge_lambda=float(data["ridge_lambda"]), use_intercept=bool(data.get("use_intercept", True)))
        if "coef" in data:
            model.coef_ = jnp.asarray(data["coef"])
            model.intercept_ = jnp.asarray(data.get("intercept", 0.0))
        return model
