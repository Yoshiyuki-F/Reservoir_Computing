"""
src/reservoir/components/readout/ridge.py
Ridge regression implementation compliant with ReadoutModule protocol.
"""
from __future__ import annotations

from typing import Dict, Any, Optional

import jax.numpy as jnp
import jax.scipy.linalg

from reservoir.core.interfaces import ReadoutModule


class RidgeRegression(ReadoutModule):
    """Ridge regression readout solved with JAX linear algebra."""

    def __init__(self, alpha: float = 1.0, use_intercept: bool = True) -> None:
        self.alpha = float(alpha)
        self.use_intercept = bool(use_intercept)
        self.coef_: Optional[jnp.ndarray] = None
        self.intercept_: Optional[jnp.ndarray] = None
        self.input_dim_: Optional[int] = None

    def fit(self, states: jnp.ndarray, targets: jnp.ndarray) -> "RidgeRegression":
        X = jnp.asarray(states, dtype=jnp.float64)
        y = jnp.asarray(targets, dtype=jnp.float64)
        if X.ndim != 2:
            raise ValueError(f"States must be 2D, got {X.shape}")
        y_is_1d = y.ndim == 1
        if y_is_1d:
            y = y[:, None]
        n_samples, n_features = X.shape
        self.input_dim_ = n_features
        X_train = jnp.concatenate([jnp.ones((n_samples, 1)), X], axis=1) if self.use_intercept else X
        XT = X_train.T
        XTX = XT @ X_train
        XTy = XT @ y
        diag_idx = jnp.diag_indices(XTX.shape[0])
        XTX = XTX.at[diag_idx].add(self.alpha)
        try:
            w = jax.scipy.linalg.solve(XTX, XTy, assume_a="pos")
        except Exception:
            w = jnp.linalg.solve(XTX, XTy)
        if self.use_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = jnp.zeros(w.shape[1], dtype=w.dtype)
            self.coef_ = w
        if y_is_1d:
            self.coef_ = self.coef_.ravel()
            self.intercept_ = self.intercept_.ravel()
        return self

    def predict(self, states: jnp.ndarray) -> jnp.ndarray:
        if self.coef_ is None:
            raise RuntimeError("RidgeRegression is not fitted yet.")
        X = jnp.asarray(states, dtype=jnp.float64)
        return X @ self.coef_ + self.intercept_ if self.use_intercept else X @ self.coef_

    def to_dict(self) -> Dict[str, Any]:
        data = {"alpha": self.alpha, "use_intercept": self.use_intercept}
        if self.coef_ is not None:
            data["coef"] = jnp.asarray(self.coef_).tolist()
            data["intercept"] = jnp.asarray(self.intercept_).tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RidgeRegression":
        model = cls(alpha=float(data.get("alpha", 1.0)), use_intercept=bool(data.get("use_intercept", True)))
        if "coef" in data:
            model.coef_ = jnp.asarray(data["coef"], dtype=jnp.float64)
            model.intercept_ = jnp.asarray(data.get("intercept", 0.0), dtype=jnp.float64)
        return model

