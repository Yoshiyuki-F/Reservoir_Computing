"""/home/yoshi/PycharmProjects/Reservoir/src/core_lib/components/readout/jax_ridge.py
JAX-native ridge regression readout."""

from __future__ import annotations

from typing import Optional, Sequence, List, Dict, Callable

import jax.numpy as jnp
from jax.scipy import linalg

from .base import BaseReadout, ReadoutResult


def _metric_fn(classification: bool) -> Callable[[jnp.ndarray, jnp.ndarray], float]:
    if classification:
        def accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
            true_labels = jnp.argmax(y_true, axis=1)
            pred_labels = jnp.argmax(y_pred, axis=1)
            return float(jnp.mean(pred_labels == true_labels))

        return accuracy

    def mse(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        diff = y_true - y_pred
        return float(jnp.mean(diff * diff))

    return mse


class RidgeReadoutJAX(BaseReadout):
    """Ridge regression solved via JAX linear algebra."""

    def __init__(self, default_lambdas: Optional[Sequence[float]] = None) -> None:
        self.default_lambdas: Sequence[float] = (
            list(default_lambdas) if default_lambdas is not None else [1e-6, 1e-4, 1e-2, 1e0, 1e2]
        )
        self.weights: Optional[jnp.ndarray] = None

    def _solve(self, X: jnp.ndarray, Y: jnp.ndarray, lam: float) -> jnp.ndarray:
        XTX = X.T @ X
        XTY = X.T @ Y
        reg = lam * jnp.eye(XTX.shape[0], dtype=X.dtype)
        return linalg.solve(XTX + reg, XTY, assume_a="pos")

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
        del cv, n_folds, random_state  # not used in this simplified JAX readout
        X_arr = jnp.asarray(X, dtype=jnp.float64)
        Y_arr = jnp.asarray(Y, dtype=jnp.float64)

        lambda_candidates = (
            sorted({float(l) for l in lambdas}) if lambdas else list(self.default_lambdas)
        )
        metric = _metric_fn(classification)
        score_name = "accuracy" if classification else "MSE"
        metric_key = "val_accuracy" if classification else "val_mse"
        logs: List[Dict[str, float]] = []

        # Simple split for tuning if multiple lambdas; use all data if single lambda.
        if len(lambda_candidates) > 1 and X_arr.shape[0] > 1:
            split = int(0.9 * X_arr.shape[0])
            split = max(1, min(split, X_arr.shape[0] - 1))
            X_train, X_val = X_arr[:split], X_arr[split:]
            Y_train, Y_val = Y_arr[:split], Y_arr[split:]
        else:
            X_train = X_val = X_arr
            Y_train = Y_val = Y_arr

        best_score = None
        best_lambda = lambda_candidates[0]
        best_weights = None

        for lam in lambda_candidates:
            weights = self._solve(X_train, Y_train, lam)
            preds = X_val @ weights
            score = metric(Y_val, preds)
            logs.append({"lambda": lam, metric_key: score})
            if best_score is None or (classification and score > best_score) or (not classification and score < best_score):
                best_score = score
                best_lambda = lam
                best_weights = weights

        if best_weights is None:
            best_weights = self._solve(X_arr, Y_arr, best_lambda)
            best_score = metric(Y_arr, X_arr @ best_weights)

        self.weights = best_weights
        return ReadoutResult(
            weights=best_weights,
            best_lambda=best_lambda,
            score_name=score_name,
            score_val=float(best_score),
            logs=logs,
        )

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("Readout has not been fitted.")
        X_arr = jnp.asarray(X, dtype=jnp.float64)
        return X_arr @ self.weights
