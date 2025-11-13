"""SVD-based ridge readout implemented with JAX arrays."""

from __future__ import annotations

from typing import List, Dict, Optional, Sequence, Callable

import jax.numpy as jnp

from .base import BaseReadout, ReadoutResult


def _default_metric(classification: bool) -> Callable[[jnp.ndarray, jnp.ndarray], float]:
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


class RidgeReadoutNumpy(BaseReadout):
    """Stable ridge regression solved via SVD."""

    def __init__(self, *, default_cv: str = "holdout", default_n_folds: int = 5) -> None:
        self.default_cv = default_cv
        self.default_n_folds = default_n_folds
        self.weights: Optional[jnp.ndarray] = None

    def _solve(self, X: jnp.ndarray, Y: jnp.ndarray, lam: float) -> jnp.ndarray:
        if X.shape[1] == 0:
            raise ValueError("Design matrix has zero columns.")
        feats = X[:, :-1]
        if feats.size == 0:
            bias = jnp.mean(Y, axis=0, keepdims=False)
            return jnp.vstack([jnp.zeros((0, Y.shape[1]), dtype=Y.dtype), bias])
        U, s, Vt = jnp.linalg.svd(feats, full_matrices=False)
        UTy = U.T @ Y
        denom = s * s + lam
        filt = (s / denom)[:, None]
        weights = Vt.T @ (filt * UTy)
        bias = jnp.mean(Y - feats @ weights, axis=0, keepdims=False)
        return jnp.vstack([weights, bias])

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
        X_np = jnp.asarray(X, dtype=jnp.float64)
        Y_np = jnp.asarray(Y, dtype=jnp.float64)

        lambda_candidates = (
            sorted({float(l) for l in lambdas}) if lambdas else [1e-6, 1e-4, 1e-2, 1e0, 1e2]
        )
        cv_mode = cv or self.default_cv
        folds = n_folds or self.default_n_folds
        metric = _default_metric(classification)
        score_name = "accuracy" if classification else "MSE"
        metric_key_val = "val_accuracy" if classification else "val_mse"
        include_train_metric = not classification
        metric_key_train = "train_mse" if include_train_metric else None
        logs: List[Dict[str, float]] = []

        if cv_mode == "kfold" and X_np.shape[0] >= folds:
            fold_size = X_np.shape[0] // folds
            scores: List[float] = []
            for lam in lambda_candidates:
                fold_scores: List[float] = []
                for fold_idx in range(folds):
                    start = fold_idx * fold_size
                    end = (fold_idx + 1) * fold_size if fold_idx < folds - 1 else X_np.shape[0]
                    val_slice = slice(start, end)
                    train_indices = jnp.concatenate([jnp.arange(0, start), jnp.arange(end, X_np.shape[0])])
                    X_train = X_np[train_indices]
                    Y_train = Y_np[train_indices]
                    X_val = X_np[val_slice]
                    Y_val = Y_np[val_slice]
                    weights = self._solve(X_train, Y_train, lam)
                    preds = X_val @ weights
                    fold_scores.append(metric(Y_val, preds))
                mean_score = float(jnp.mean(jnp.array(fold_scores)))
                logs.append(
                    {
                        "lambda": lam,
                        metric_key_val: mean_score,
                    }
                )
                scores.append(mean_score)
            scores_arr = jnp.array(scores)
            if classification:
                best_score = float(jnp.max(scores_arr))
                best_lambda = max(
                    lam
                    for lam, score in zip(lambda_candidates, scores)
                    if float(score) >= best_score - 1e-12
                )
            else:
                best_idx = int(jnp.argmin(scores_arr))
                best_lambda = lambda_candidates[best_idx]
                best_score = scores[best_idx]
            final_weights = self._solve(X_np, Y_np, best_lambda)
        else:
            split = int(0.9 * X_np.shape[0])
            split = max(1, min(split, X_np.shape[0] - 1))
            X_train, X_val = X_np[:split], X_np[split:]
            Y_train, Y_val = Y_np[:split], Y_np[split:]
            scores = []
            for lam in lambda_candidates:
                weights = self._solve(X_train, Y_train, lam)
                train_score = None
                if include_train_metric:
                    train_pred = X_train @ weights
                    train_score = metric(Y_train, train_pred)
                if X_val.size:
                    eval_X = X_val
                    eval_Y = Y_val
                else:
                    eval_X = X_train
                    eval_Y = Y_train
                preds = eval_X @ weights
                val_score = metric(eval_Y, preds)
                log_entry = {
                    "lambda": lam,
                    metric_key_val: val_score,
                }
                if include_train_metric and metric_key_train:
                    log_entry[metric_key_train] = train_score
                logs.append(log_entry)
                scores.append(val_score)
            scores_arr = jnp.array(scores)
            if classification:
                best_score = float(jnp.max(scores_arr))
                best_lambda = max(
                    lam
                    for lam, score in zip(lambda_candidates, scores)
                    if float(score) >= best_score - 1e-12
                )
            else:
                best_idx = int(jnp.argmin(scores_arr))
                best_lambda = lambda_candidates[best_idx]
                best_score = scores[best_idx]
            final_weights = self._solve(X_np, Y_np, best_lambda)

        self.weights = final_weights
        return ReadoutResult(
            weights=final_weights,
            best_lambda=best_lambda,
            score_name=score_name,
            score_val=float(best_score),
            logs=logs,
        )

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("Readout has not been fitted.")
        X_np = jnp.asarray(X, dtype=jnp.float64)
        return X_np @ self.weights
